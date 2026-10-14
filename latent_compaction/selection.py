"""KVZip+ preservers and attention/farthest-key soft-slot initialization."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def capture_kvzip_scores(model: Any, context_ids: Tensor, cache: Any, preserver_fraction: float) -> Tensor:
    """Prefill once under KVZip+ and return its per-layer/head reconstruction scores."""
    if not 0 < preserver_fraction < 1:
        raise ValueError("preserver_fraction must be between zero and one")
    try:
        from kvpress import KVzipPress
    except ImportError as exc:
        raise ImportError("Install kvpress to select reconstruction-based preservers") from exc

    class CapturingPress(KVzipPress):
        scores: Tensor | None = None

        def compress_post(self, fitted_model: Any) -> None:
            if self.score_val is None:
                raise RuntimeError("KVZip+ did not produce reconstruction scores")
            self.scores = self.score_val.detach().clone()
            super().compress_post(fitted_model)

    press = CapturingPress(
        compression_ratio=1.0 - preserver_fraction,
        kvzip_plus_normalization=True,
    )
    with torch.no_grad(), press(model):
        model.model(input_ids=context_ids, past_key_values=cache)
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None
    if press.scores is None:
        raise RuntimeError("KVZip+ score capture failed")
    return press.scores


def select_preservers(scores: Tensor, average_per_head: int) -> Tensor:
    """Select a fixed global budget over layer, KV head, and source position."""
    if scores.ndim != 4 or scores.shape[1] != 1:
        raise ValueError("scores must have shape [layers, 1, KV heads, positions]")
    layers, _, heads, positions = scores.shape
    if not 0 < average_per_head < positions:
        raise ValueError("invalid preserver budget")
    total = layers * heads * average_per_head
    indices = scores.float().reshape(-1).topk(total).indices
    mask = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    mask[indices] = True
    return mask.view_as(scores)


def gather_preservers(keys: Tensor, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Pad ragged per-head selections; the returned mask excludes padding."""
    if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
        raise ValueError("keys and values must match [1, KV heads, positions, dimension]")
    if mask.shape != keys.shape[1:3]:
        raise ValueError("mask must have shape [KV heads, positions]")
    counts = mask.sum(-1)
    width = int(counts.max().item())
    out_keys = keys.new_zeros((1, keys.shape[1], width, keys.shape[-1]))
    out_values = values.new_zeros(out_keys.shape)
    valid = torch.zeros((keys.shape[1], width), dtype=torch.bool, device=keys.device)
    for head in range(keys.shape[1]):
        selected = mask[head].nonzero(as_tuple=True)[0]
        count = selected.numel()
        out_keys[0, head, :count] = keys[0, head].index_select(0, selected)
        out_values[0, head, :count] = values[0, head].index_select(0, selected)
        valid[head, :count] = True
    return out_keys, out_values, valid


@torch.no_grad()
def initialize_soft_slots(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    *,
    slots: int,
    scaling: float,
    anchor_fraction: float = 0.25,
    query_chunk: int = 64,
) -> tuple[Tensor, Tensor]:
    """Seed soft slots with attention anchors, then greedy farthest-key coverage."""
    if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
        raise ValueError("keys and values must match [1, KV heads, positions, dimension]")
    if queries.ndim != 4 or queries.shape[0] != 1 or queries.shape[-1] != keys.shape[-1]:
        raise ValueError("queries must have shape [1, query heads, queries, dimension]")
    if not 1 <= slots <= keys.shape[-2] or not 0 < anchor_fraction <= 1:
        raise ValueError("invalid slot count or anchor fraction")
    heads = keys.shape[1]
    groups = queries.shape[1] // heads
    if groups * heads != queries.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    anchor_count = max(1, int(slots * anchor_fraction))
    chosen_keys, chosen_values = [], []
    for head in range(heads):
        points = keys[0, head].float()
        head_queries = queries[0, head * groups:(head + 1) * groups].float()
        importance = torch.zeros(points.shape[0], device=points.device)
        for start in range(0, head_queries.shape[1], query_chunk):
            q = head_queries[:, start:start + query_chunk]
            weights = torch.softmax(torch.einsum("gqd,nd->gqn", q, points) * scaling, -1)
            importance = torch.maximum(importance, weights.amax((0, 1)))
        importance[0] = torch.inf  # Preserve the conventional first-position sink.
        chosen = importance.topk(anchor_count).indices.tolist()
        selected = torch.tensor(chosen, device=points.device)
        distances = torch.cdist(points, points.index_select(0, selected)).square().amin(-1)
        distances[selected] = -1
        for _ in range(anchor_count, slots):
            next_index = int(distances.argmax())
            chosen.append(next_index)
            distance = (points - points[next_index]).square().sum(-1)
            distances = torch.minimum(distances, distance)
            distances[next_index] = -1
        indices = torch.tensor(chosen, device=points.device)
        chosen_keys.append(keys[0, head].index_select(0, indices))
        chosen_values.append(values[0, head].index_select(0, indices))
    return torch.stack(chosen_keys).unsqueeze(0), torch.stack(chosen_values).unsqueeze(0)
