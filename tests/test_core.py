"""Small, model-independent checks of the public method path."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from latent_compaction import CompactedMemory, SOURCE_PROMPTS, install_attention_adapter, resolve_model_name
from latent_compaction.models import generate_from_memory
from latent_compaction.selection import initialize_soft_slots, select_preservers
from latent_compaction.methods.latent import FitConfig, fit_document, load_memory, save_memory


class TinyTokenizer:
    def __call__(self, text, return_tensors, add_special_tokens):
        del return_tensors, add_special_tokens
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) % 127 + 1 for c in text]]))

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(map(str, ids))


class CoreTests(unittest.TestCase):
    def test_prompt_bank_and_global_budget(self):
        self.assertEqual(len(SOURCE_PROMPTS), 480)
        self.assertEqual(len(set(SOURCE_PROMPTS)), 480)
        scores = torch.arange(40.0).reshape(2, 1, 2, 10)
        mask = select_preservers(scores, 3)
        self.assertEqual(int(mask.sum()), 2 * 2 * 3)

    def test_anchors_and_coverage_are_distinct(self):
        keys = torch.arange(8.0).reshape(1, 1, 8, 1)
        values = keys.clone()
        queries = torch.ones((1, 1, 2, 1))
        selected, _ = initialize_soft_slots(keys, values, queries, slots=4, scaling=1.0)
        self.assertEqual(selected.shape, (1, 1, 4, 1))
        self.assertIn(0.0, selected.flatten().tolist())
        self.assertEqual(len(set(selected.flatten().tolist())), 4)

    def _fit_tiny_model(self, family):
        torch.manual_seed(7)
        config_class, model_class = (
            (Qwen3Config, Qwen3ForCausalLM) if family == "qwen3"
            else (LlamaConfig, LlamaForCausalLM)
        )
        config = config_class(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=256,
        )
        model = model_class(config).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            native_logits = model(input_ids=input_ids, use_cache=True).logits
        install_attention_adapter(model)
        with torch.no_grad():
            adapted_logits = model(input_ids=input_ids, use_cache=True).logits
        torch.testing.assert_close(adapted_logits, native_logits, rtol=0, atol=0)
        tokenizer = TinyTokenizer()

        def fake_kvzip(fitted_model, context_ids, cache, preserver_fraction):
            self.assertGreater(preserver_fraction, 0)
            with torch.no_grad():
                fitted_model.model(input_ids=context_ids, past_key_values=cache)
            return torch.rand((2, 1, 2, context_ids.shape[-1]))

        with patch("latent_compaction.methods.latent.capture_kvzip_scores", fake_kvzip):
            memories, length = fit_document(
                model, tokenizer, "This is a sample document with several facts.",
                FitConfig(budget=8, soft_slots=2, stage1_steps=1, stage2_steps=1,
                          rollout_tokens=2, stage1_queries=8, stage1_batch_size=4),
                prompts=("Summarize. Answer:", "Describe. Answer:"),
            )
        self.assertEqual(len(memories), 2)
        self.assertTrue(all(isinstance(m, CompactedMemory) for m in memories.values()))
        self.assertTrue(all(m.preserver_keys.requires_grad for m in memories.values()))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "memory.pt"
            save_memory(artifact, memories, length, model_id=family)
            with self.assertRaisesRegex(ValueError, "different model"):
                load_memory(artifact, "cpu", expected_model_id="other-model")
            restored, restored_length = load_memory(artifact, "cpu", expected_model_id=family)
            self.assertEqual(restored_length, length)
            answer = generate_from_memory(model, tokenizer, restored, length, "Question? Answer:", max_new_tokens=2)
            self.assertIsInstance(answer, str)

    def test_two_stages_and_inference_on_tiny_qwen(self):
        self._fit_tiny_model("qwen3")

    def test_two_stages_and_inference_on_tiny_llama(self):
        self._fit_tiny_model("llama")

    def test_model_aliases(self):
        self.assertEqual(resolve_model_name("minicpm5-2b"), "openbmb/MiniCPM5-2B")
        self.assertEqual(resolve_model_name("llama-3.2-3b-instruct"), "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(resolve_model_name("/checkpoints/my-model"), "/checkpoints/my-model")


if __name__ == "__main__":
    unittest.main()
