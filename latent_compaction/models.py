"""Qwen3 and Llama-family cache adapters with a compacted document prefix."""

from __future__ import annotations

import types
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_rotary
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_rotary

from .memory import CompactedMemory


MODEL_ALIASES = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "minicpm5-1b": "openbmb/MiniCPM5-1B",
    "minicpm5-2b": "openbmb/MiniCPM5-2B",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
}


def resolve_model_name(model_name: str) -> str:
    """Expand convenience aliases; full Hub IDs and local paths pass through."""
    return MODEL_ALIASES.get(model_name.lower(), model_name)


class DocumentCache(DynamicCache):
    """Exact suffix cache with an optional logical compacted document prefix."""

    def __init__(self, config: Any = None) -> None:
        super().__init__(config=config)
        self.document_tokens = 0
        self.memories: dict[int, CompactedMemory] = {}
        self.capture_queries = False
        self.queries: dict[int, list[Tensor]] = {}

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.document_tokens + super().get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        length, offset = super().get_mask_sizes(query_length, layer_idx)
        return self.document_tokens + length, offset

    def fork(self, *, capture_queries: bool = False) -> DocumentCache:
        clone = DocumentCache()
        for layer_idx, layer in enumerate(self.layers):
            keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
            if isinstance(keys, Tensor) and isinstance(values, Tensor) and keys.numel():
                clone.update(keys, values, layer_idx)
        clone.document_tokens = self.document_tokens
        clone.memories = self.memories.copy()
        clone.capture_queries = capture_queries
        return clone


def _causal_local_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    attention_mask: Tensor | None,
    *,
    scaling: float,
    groups: int,
) -> tuple[Tensor, Tensor]:
    keys = keys.repeat_interleave(groups, dim=1)
    values = values.repeat_interleave(groups, dim=1)
    logits = queries.float() @ keys.float().transpose(-1, -2) * scaling
    query_length, key_length = logits.shape[-2:]
    past_length = key_length - query_length
    q_positions = torch.arange(query_length, device=logits.device) + past_length
    k_positions = torch.arange(key_length, device=logits.device)
    logits = logits.masked_fill(
        (k_positions[None, :] > q_positions[:, None])[None, None],
        torch.finfo(logits.dtype).min,
    )
    if attention_mask is not None:
        local_mask = attention_mask[..., -key_length:]
        if local_mask.dtype == torch.bool:
            logits = logits.masked_fill(~local_mask, torch.finfo(logits.dtype).min)
        else:
            logits = logits + local_mask.float()
    return torch.softmax(logits, -1) @ values.float(), torch.logsumexp(logits, -1)


def _forward(
    self: nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor | None]:
    cache = past_key_values
    if not isinstance(cache, DocumentCache) or (
        not cache.capture_queries and self.layer_idx not in cache.memories
    ):
        return self._latent_compaction_original_forward(
            hidden_states, position_embeddings, attention_mask, past_key_values=cache, **kwargs
        )

    shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    queries = self.q_proj(hidden_states).view(shape)
    keys = self.k_proj(hidden_states).view(shape)
    if hasattr(self, "q_norm"):
        queries = self.q_norm(queries)
    if hasattr(self, "k_norm"):
        keys = self.k_norm(keys)
    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    rotary = qwen3_rotary if self.__class__.__name__ == "Qwen3Attention" else llama_rotary
    queries, keys = rotary(queries, keys, *position_embeddings)
    if cache.capture_queries:
        cache.queries.setdefault(self.layer_idx, []).append(queries.detach().cpu())

    if self.layer_idx not in cache.memories:
        return self._latent_compaction_original_forward(
            hidden_states, position_embeddings, attention_mask, past_key_values=cache, **kwargs
        )

    keys, values = cache.update(keys, values, self.layer_idx)
    memory = cache.memories[self.layer_idx]
    document_output, document_log_z = memory(queries)
    recent_output, recent_log_z = _causal_local_attention(
        queries, keys, values, attention_mask, scaling=float(self.scaling),
        groups=self.num_key_value_groups,
    )
    total_log_z = torch.logaddexp(document_log_z, recent_log_z)
    output = (
        torch.exp(document_log_z - total_log_z)[..., None] * document_output.float()
        + torch.exp(recent_log_z - total_log_z)[..., None] * recent_output
    )
    output = output.to(hidden_states.dtype).transpose(1, 2).contiguous()
    output = output.reshape(*hidden_states.shape[:-1], -1)
    return self.o_proj(output), None


def install_attention_adapter(model: nn.Module) -> None:
    """Patch Qwen3 or Llama attention; ordinary full-cache calls remain native."""
    for layer in model.model.layers:
        attention = layer.self_attn
        if attention.__class__.__name__ not in {"Qwen3Attention", "LlamaAttention"}:
            raise TypeError("Latent Compaction expects Qwen3Attention or LlamaAttention")
        if not hasattr(attention, "_latent_compaction_original_forward"):
            attention._latent_compaction_original_forward = attention.forward
            attention.forward = types.MethodType(_forward, attention)


def load_model(model_name: str, device: str = "cuda") -> tuple[Any, Any]:
    model_name = resolve_model_name(model_name)
    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(target).eval()
    if model.config.model_type not in {"qwen3", "llama"}:
        raise TypeError("Latent Compaction supports Qwen3 and Llama-family checkpoints")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    install_attention_adapter(model)
    return model, tokenizer


@torch.no_grad()
def generate_from_memory(
    model: Any,
    tokenizer: Any,
    memories: dict[int, CompactedMemory],
    document_tokens: int,
    prompt: str,
    *,
    max_new_tokens: int = 128,
) -> str:
    device = next(model.parameters()).device
    token_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    cache = DocumentCache(model.config)
    cache.document_tokens = document_tokens
    cache.memories = memories
    logits = model(input_ids=token_ids, past_key_values=cache, use_cache=True).logits[:, -1]
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_token = logits.argmax(-1, keepdim=True)
        token = int(next_token.item())
        if token in eos_ids:
            break
        generated.append(token)
        logits = model(input_ids=next_token, past_key_values=cache, use_cache=True).logits[:, -1]
    return tokenizer.decode(generated, skip_special_tokens=True)
