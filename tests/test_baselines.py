"""Baseline cache shape and finite-output checks on a local tiny reader."""
from pathlib import Path
import sys
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from latent_compaction.models import install_attention_adapter
from test_core import TinyTokenizer
from latent_compaction.runner import construct


def test_portable_baseline_paths():
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512)
    model = Qwen3ForCausalLM(config).eval()
    model.requires_grad_(False)
    install_attention_adapter(model)
    for method in ('full', 'snapkv', 'pyramidkv', 'attention_matching'):
        exact, memories, length = construct(model, TinyTokenizer(), 'Synthetic context with sufficient length. '*3, method, 40)
        assert length > 40
        if memories is not None:
            assert len(memories) == 2
            for memory in memories.values():
                output, normalizer = memory(torch.ones(1, 4, 1, 16))
                assert torch.isfinite(output).all() and torch.isfinite(normalizer).all()


def test_eviction_small_and_capped_budgets():
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512)
    model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    install_attention_adapter(model)
    for method in ('snapkv', 'pyramidkv'):
        for budget in (4, 32, 40, 100):
            exact, memories, length = construct(model, TinyTokenizer(), 'Synthetic context with sufficient length. '*3, method, budget)
            assert sum(m.logical_slots_per_head for m in memories.values()) == 2 * budget
            assert all(m.preserver_keys.shape[-2] <= length for m in memories.values())
