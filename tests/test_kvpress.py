"""Exercise the real score-capture hook without external models or downloads."""
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from latent_compaction.models import DocumentCache, install_attention_adapter
from latent_compaction.selection import capture_kvzip_scores


def test_real_kvzip_capture():
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512)
    model = Qwen3ForCausalLM(config).eval()
    install_attention_adapter(model)
    for p in model.parameters():
        p.requires_grad_(False)
    class Tokenizer:
        chat_template = None
        def encode(self, text, return_tensors=None, **kwargs):
            return torch.tensor([[1, 2, 3]]) if return_tensors else [1, 2, 3]
    # KVZip loads a tokenizer internally; substitute only its remote tokenizer I/O.
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()):
        cache = DocumentCache(config)
        scores = capture_kvzip_scores(model, torch.arange(1, 33)[None], cache, .5)
    assert scores.shape == (2, 1, 2, 32)
    assert torch.isfinite(scores).all()
    assert all(layer.keys.shape[-2] == 32 for layer in cache.layers)
