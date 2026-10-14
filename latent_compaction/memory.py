"""Biased attention over KVZip+ preservers and trainable soft slots."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CompactedMemory(nn.Module):
    """Per-layer document memory; padded preserver rows are masked out."""

    def __init__(
        self,
        preserver_keys: Tensor,
        preserver_values: Tensor,
        preserver_mask: Tensor,
        soft_keys: Tensor,
        soft_values: Tensor,
        *,
        query_heads: int,
        scaling: float,
    ) -> None:
        super().__init__()
        tensors = (preserver_keys, preserver_values, soft_keys, soft_values)
        if any(t.ndim != 4 or t.shape[0] != 1 for t in tensors):
            raise ValueError("KV tensors must have shape [1, KV heads, slots, dimension]")
        if preserver_keys.shape != preserver_values.shape or soft_keys.shape != soft_values.shape:
            raise ValueError("key/value shapes must match")
        if preserver_keys.shape[1] != soft_keys.shape[1] or preserver_keys.shape[-1] != soft_keys.shape[-1]:
            raise ValueError("preserver and soft-slot head shapes must match")
        if preserver_mask.shape != preserver_keys.shape[1:3] or preserver_mask.dtype != torch.bool:
            raise ValueError("preserver_mask must be bool [KV heads, slots]")
        if query_heads % preserver_keys.shape[1]:
            raise ValueError("query heads must be divisible by KV heads")
        self.preserver_keys = nn.Parameter(preserver_keys[0].float().clone())
        self.preserver_values = nn.Parameter(preserver_values[0].float().clone())
        self.soft_keys = nn.Parameter(soft_keys[0].float().clone())
        self.soft_values = nn.Parameter(soft_values[0].float().clone())
        self.preserver_bias = nn.Parameter(torch.zeros_like(preserver_mask, dtype=torch.float32))
        self.soft_bias = nn.Parameter(torch.zeros(soft_keys.shape[1:3], device=soft_keys.device))
        self.register_buffer("preserver_mask", preserver_mask.clone())
        self.query_heads = query_heads
        self.groups = query_heads // preserver_keys.shape[1]
        self.scaling = scaling
        self.stage_one()

    def stage_one(self) -> None:
        """Fit soft K/V and biases while preservers stay fixed."""
        for p in (self.preserver_keys, self.preserver_values, self.preserver_bias):
            p.requires_grad_(False)
        self.soft_keys.requires_grad_(True)
        self.soft_values.requires_grad_(True)
        self.soft_bias.requires_grad_(True)

    def stage_two(self) -> None:
        """Jointly refine preserver K/V, soft K/V, and attention biases."""
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, queries: Tensor) -> tuple[Tensor, Tensor]:
        if queries.ndim != 4 or queries.shape[1] != self.query_heads:
            raise ValueError("queries must have shape [batch, query heads, positions, dimension]")
        keys = torch.cat((self.preserver_keys, self.soft_keys), dim=1)
        values = torch.cat((self.preserver_values, self.soft_values), dim=1)
        biases = torch.cat((self.preserver_bias, self.soft_bias), dim=1)
        valid = torch.cat((self.preserver_mask, torch.ones_like(self.soft_bias, dtype=torch.bool)), dim=1)
        keys = keys.repeat_interleave(self.groups, dim=0)
        values = values.repeat_interleave(self.groups, dim=0)
        biases = biases.repeat_interleave(self.groups, dim=0)
        valid = valid.repeat_interleave(self.groups, dim=0)
        logits = torch.einsum("bhqd,hnd->bhqn", queries.float(), keys) * self.scaling
        logits = logits + biases[None, :, None, :]
        logits = logits.masked_fill(~valid[None, :, None, :], torch.finfo(logits.dtype).min)
        log_normalizer = torch.logsumexp(logits, dim=-1)
        output = torch.einsum("bhqn,hnd->bhqd", torch.softmax(logits, -1), values)
        return output.to(queries.dtype), log_normalizer

    @property
    def logical_slots_per_head(self) -> float:
        return self.preserver_mask.sum().item() / self.preserver_mask.shape[0] + self.soft_keys.shape[1]


def selected_memory(model, layer_index, keys, values, mask=None, bias=None):
    """Wrap selected KV entries for the shared continuation-attention adapter."""
    if mask is None:
        mask = torch.ones(keys.shape[1:3], dtype=torch.bool, device=keys.device)
    empty = keys[..., :0, :]
    memory = CompactedMemory(keys, values, mask, empty, empty,
        query_heads=model.config.num_attention_heads,
        scaling=float(model.model.layers[layer_index].self_attn.scaling)).to(keys.device)
    if bias is not None:
        with torch.no_grad():
            memory.preserver_bias.copy_(bias)
    return memory.requires_grad_(False).eval()
