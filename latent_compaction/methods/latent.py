"""Source-only teacher rollouts and the two document-specific training stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from ..memory import CompactedMemory
from ..prompts import SOURCE_PROMPTS
from ..models import DocumentCache
from ..selection import (
    capture_kvzip_scores,
    gather_preservers,
    initialize_soft_slots,
    select_preservers,
)


@dataclass(frozen=True)
class FitConfig:
    budget: int
    soft_slots: int
    stage1_steps: int = 1000
    stage2_steps: int = 1000
    rollout_tokens: int = 24
    stage1_queries: int = 4096
    stage1_batch_size: int = 128
    stage2_batch_size: int = 4
    stage1_learning_rate: float = 0.003
    stage2_learning_rate: float = 0.0001
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.stage1_queries, self.stage1_batch_size, self.stage2_batch_size) < 1:
            raise ValueError("query and batch sizes must be positive")
        if min(self.stage1_learning_rate, self.stage2_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if self.stage1_queries < 2:
            raise ValueError("stage1_queries must be at least two for normalizer statistics")
        if not 0 < self.soft_slots < self.budget:
            raise ValueError("soft_slots must be positive and smaller than the total budget")
        if min(self.stage1_steps, self.stage2_steps) < 0 or self.rollout_tokens < 1:
            raise ValueError("stage steps must be nonnegative and rollout length positive")


@dataclass
class TeacherRollout:
    prompt_ids: Tensor
    generated_ids: list[int]
    top_indices: list[Tensor]
    top_values: list[Tensor]
    log_normalizers: list[Tensor]


@torch.no_grad()
def collect_teacher_rollouts(
    model: Any,
    tokenizer: Any,
    full_cache: DocumentCache,
    *,
    prompts: Sequence[str] = SOURCE_PROMPTS,
    max_tokens: int = 24,
) -> tuple[dict[int, Tensor], list[TeacherRollout]]:
    """Probe each document with the fixed generic bank; no evaluation queries."""
    if not prompts:
        raise ValueError("at least one source prompt is required")
    device = next(model.parameters()).device
    query_lists: dict[int, list[Tensor]] = {}
    records: list[TeacherRollout] = []
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    for prompt in prompts:
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        cache = full_cache.fork(capture_queries=True)
        logits = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True).logits[:, -1]
        generated: list[int] = []
        top_indices: list[Tensor] = []
        top_values: list[Tensor] = []
        log_normalizers: list[Tensor] = []
        for _ in range(max_tokens):
            scores = logits.float()
            values, indices = scores.topk(min(20, scores.shape[-1]), dim=-1)
            best = scores.argmax(-1, keepdim=True)
            absent = ~(indices == best).any(-1, keepdim=True)
            indices[..., -1:] = torch.where(absent, best, indices[..., -1:])
            values = scores.gather(-1, indices)
            top_indices.append(indices.squeeze(0).cpu())
            top_values.append(values.squeeze(0).cpu())
            log_normalizers.append(scores.logsumexp(-1).squeeze(0).cpu())
            token = int(scores.argmax(-1).item())
            generated.append(token)
            if token in eos_ids:
                break
            input_ids = torch.tensor([[token]], device=device)
            logits = model(input_ids=input_ids, past_key_values=cache, use_cache=True).logits[:, -1]
        for layer, chunks in cache.queries.items():
            query_lists.setdefault(layer, []).extend(chunks)
        records.append(TeacherRollout(prompt_ids.cpu(), generated, top_indices, top_values, log_normalizers))
    return {layer: torch.cat(chunks, dim=-2) for layer, chunks in query_lists.items()}, records


@torch.no_grad()
def exact_attention_targets(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    scaling: float,
    chunk_size: int = 16,
) -> tuple[Tensor, Tensor]:
    groups = queries.shape[1] // keys.shape[1]
    if groups * keys.shape[1] != queries.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    repeated_keys = keys.float().repeat_interleave(groups, dim=1)
    repeated_values = values.float().repeat_interleave(groups, dim=1)
    outputs, normalizers = [], []
    for start in range(0, queries.shape[-2], chunk_size):
        q = queries[..., start:start + chunk_size, :].float()
        scores = q @ repeated_keys.transpose(-1, -2) * scaling
        outputs.append(torch.softmax(scores, -1) @ repeated_values)
        normalizers.append(scores.logsumexp(-1))
    return torch.cat(outputs, -2), torch.cat(normalizers, -1)


def fit_attention_stage(
    memory: CompactedMemory,
    queries: Tensor,
    full_keys: Tensor,
    full_values: Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    """Match full-cache attention values and log normalizers per layer."""
    memory.stage_one()
    device = full_keys.device
    queries = queries.to(device)
    targets, target_log_z = exact_attention_targets(
        queries, full_keys, full_values, scaling=memory.scaling
    )
    value_mean = targets.mean((-2, -1), keepdim=True)
    value_scale = targets.std((-2, -1), keepdim=True).clamp_min(1e-4)
    log_mean = target_log_z.mean(-1, keepdim=True)
    log_scale = target_log_z.std(-1, keepdim=True).clamp_min(1e-4)
    optimizer = torch.optim.AdamW((memory.soft_keys, memory.soft_values, memory.soft_bias), lr=learning_rate, weight_decay=0)
    sampling_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=sampling_device).manual_seed(seed)
    for _ in range(steps):
        indices = torch.randint(queries.shape[-2], (min(batch_size, queries.shape[-2]),), device=sampling_device, generator=generator).to(device)
        q = queries.index_select(-2, indices)
        value, log_z = memory(q)
        expected = targets.index_select(-2, indices)
        expected_log_z = target_log_z.index_select(-1, indices)
        loss = F.mse_loss((value.float() - value_mean) / value_scale, (expected - value_mean) / value_scale)
        loss = loss + F.mse_loss((log_z - log_mean) / log_scale, (expected_log_z - log_mean) / log_scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((memory.soft_keys, memory.soft_values, memory.soft_bias), 1.0)
        optimizer.step()


def grouped_teacher_kl(student_logits: Tensor, record: TeacherRollout) -> Tensor:
    """KL over the teacher top 20 tokens and one aggregate tail category."""
    device = student_logits.device
    selected = torch.stack(record.top_indices).to(device)
    teacher_log_p = torch.stack(record.top_values).to(device) - torch.stack(record.log_normalizers).to(device)[:, None]
    teacher_p = teacher_log_p.exp()
    student_log_p = F.log_softmax(student_logits.float(), -1)
    selected_student = student_log_p.gather(-1, selected)
    teacher_tail = (1 - teacher_p.sum(-1)).clamp_min(1e-8)
    student_tail = (1 - selected_student.exp().sum(-1)).clamp_min(1e-8)
    kl = (teacher_p * (teacher_log_p - selected_student)).sum(-1)
    kl = kl + teacher_tail * (teacher_tail.log() - student_tail.log())
    top_one = torch.tensor(record.generated_ids, device=device)
    return kl.mean() + F.cross_entropy(student_logits.float(), top_one)


def fit_distillation_stage(
    model: Any,
    memories: dict[int, CompactedMemory],
    document_tokens: int,
    rollouts: Sequence[TeacherRollout],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    """Jointly refine both KV groups and their biases using source-only logits."""
    if not rollouts:
        raise ValueError("teacher rollouts cannot be empty")
    for memory in memories.values():
        memory.stage_two()
    parameters = [p for memory in memories.values() for p in memory.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for index in torch.randint(len(rollouts), (batch_size,), generator=generator).tolist():
            record = rollouts[index]
            prefix = record.generated_ids[:-1]
            input_ids = torch.cat((record.prompt_ids.flatten(), torch.tensor(prefix, dtype=torch.long))).unsqueeze(0).to(device)
            cache = DocumentCache(model.config)
            cache.document_tokens = document_tokens
            cache.memories = memories
            all_logits = model(input_ids=input_ids, past_key_values=cache, use_cache=True).logits[0]
            predictions = all_logits[-len(record.generated_ids):]
            (grouped_teacher_kl(predictions, record) / batch_size).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()


def fit_document(
    model: Any,
    tokenizer: Any,
    document: str,
    config: FitConfig,
    *,
    prompts: Sequence[str] = SOURCE_PROMPTS,
) -> tuple[dict[int, CompactedMemory], int]:
    """Construct and optimize one reusable compacted document cache."""
    device = next(model.parameters()).device
    context_ids = tokenizer(document, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    document_tokens = context_ids.shape[-1]
    torch.manual_seed(config.seed)
    if document_tokens == 0:
        raise ValueError("document cannot be empty")
    if config.budget >= document_tokens:
        raise ValueError("budget must be smaller than the tokenized document")
    full = DocumentCache(model.config)
    full.capture_queries = False
    preservers = config.budget - config.soft_slots
    scores = capture_kvzip_scores(model, context_ids, full, preservers / document_tokens)
    selected = select_preservers(scores, preservers)
    query_banks, rollouts = collect_teacher_rollouts(
        model, tokenizer, full, prompts=prompts, max_tokens=config.rollout_tokens
    )
    memories: dict[int, CompactedMemory] = {}
    for layer_index, layer in enumerate(full.layers):
        queries = query_banks[layer_index]
        if queries.shape[-2] > config.stage1_queries:
            sample = torch.linspace(0, queries.shape[-2] - 1, config.stage1_queries).long()
            queries = queries.index_select(-2, sample)
        queries = queries.to(device)
        preserver_keys, preserver_values, mask = gather_preservers(
            layer.keys, layer.values, selected[layer_index, 0]
        )
        attention = model.model.layers[layer_index].self_attn
        soft_keys, soft_values = initialize_soft_slots(
            layer.keys, layer.values, queries, slots=config.soft_slots,
            scaling=float(attention.scaling),
        )
        memory = CompactedMemory(
            preserver_keys, preserver_values, mask, soft_keys, soft_values,
            query_heads=attention.config.num_attention_heads,
            scaling=float(attention.scaling),
        ).to(device)
        fit_attention_stage(
            memory, queries, layer.keys, layer.values,
            steps=config.stage1_steps, batch_size=config.stage1_batch_size,
            learning_rate=config.stage1_learning_rate, seed=config.seed + layer_index,
        )
        memories[layer_index] = memory
    fit_distillation_stage(
        model, memories, document_tokens, rollouts,
        steps=config.stage2_steps, batch_size=config.stage2_batch_size,
        learning_rate=config.stage2_learning_rate, seed=config.seed,
    )
    return memories, document_tokens


def save_memory(
    path: str | Path,
    memories: dict[int, CompactedMemory],
    document_tokens: int,
    *,
    model_id: str,
) -> None:
    """Save only the compacted KV parameters, without the original document."""
    layers = []
    for index in sorted(memories):
        memory = memories[index]
        layers.append({
            "layer": index,
            "query_heads": memory.query_heads,
            "scaling": memory.scaling,
            "state": {name: tensor.detach().cpu() for name, tensor in memory.state_dict().items()},
        })
    torch.save({"format": 2, "model_id": model_id, "document_tokens": document_tokens, "layers": layers}, path)


def load_memory(
    path: str | Path,
    device: str | torch.device,
    *,
    expected_model_id: str | None = None,
) -> tuple[dict[int, CompactedMemory], int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") not in {1, 2}:
        raise ValueError("unsupported compacted cache artifact format")
    if expected_model_id is not None and payload.get("model_id") != expected_model_id:
        raise ValueError("compacted cache was fitted for a different model")
    memories = {}
    for row in payload["layers"]:
        state = row["state"]
        memory = CompactedMemory(
            state["preserver_keys"].unsqueeze(0), state["preserver_values"].unsqueeze(0),
            state["preserver_mask"], state["soft_keys"].unsqueeze(0), state["soft_values"].unsqueeze(0),
            query_heads=row["query_heads"], scaling=row["scaling"],
        )
        memory.load_state_dict(state)
        memory.stage_two()
        memory.eval().to(device)
        memories[row["layer"]] = memory
    return memories, payload["document_tokens"]
