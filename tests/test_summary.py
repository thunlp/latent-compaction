"""Text-summary budgets, source-only generation, and shared-cache evaluation."""
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

from latent_compaction.models import install_attention_adapter
from latent_compaction.methods.summary import clip_summary, summarize_document


class CharacterTokenizer:
    def __init__(self):
        self.messages = []

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(int(i)) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append((messages, kwargs))
        return messages[0]['content']

    def __call__(self, text, **kwargs):
        return BatchEncoding({'input_ids': torch.tensor([self.encode(text)])})


class SummaryModel(nn.Module):
    def __init__(self, answer):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.answer = answer
        self.calls = []

    def generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        return torch.cat((input_ids, torch.tensor([[ord(c) for c in self.answer]])), dim=-1)


@pytest.mark.parametrize('budget', [5, 300])
def test_source_only_summary_and_generation_settings(budget):
    tok = CharacterTokenizer()
    model = SummaryModel('Silver Ridge opened in 1998.')
    result = summarize_document(model, tok, 'Source document only.', budget)
    messages, options = tok.messages[0]
    assert messages[0]['content'].endswith('DOCUMENT:\nSource document only.')
    assert len(messages) == 1 and options['enable_thinking'] is False
    assert model.calls == [{'max_new_tokens': max(512, budget*2), 'do_sample': False}]
    assert 0 < result['summary_tokens'] <= budget
    assert result['summary'] == model.answer[:budget]
    assert result['heldout_queries_sent'] is False


def test_clip_handles_decode_reencode_growth():
    class ExpandingTokenizer(CharacterTokenizer):
        def decode(self, ids, **kwargs):
            return super().decode(ids) + '!'
    text, count = clip_summary(ExpandingTokenizer(), 'abcdef', 4)
    assert text == 'abc!' and count == 4


def test_empty_and_invalid_summaries_fail():
    with pytest.raises(ValueError, match='empty'):
        clip_summary(CharacterTokenizer(), '', 10)
    with pytest.raises(ValueError, match='positive'):
        clip_summary(CharacterTokenizer(), 'a', 0)


def test_baseline_reuses_one_summary_for_two_questions(tmp_path, monkeypatch):
    from latent_compaction import runner as run_baseline
    tok = CharacterTokenizer()
    config = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=1024)
    model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    install_attention_adapter(model)
    generated = []
    def generate(input_ids, **kwargs):
        generated.append(kwargs)
        return torch.cat((input_ids, torch.tensor([tok.encode('Silver Ridge in 1998.')])), dim=-1)
    monkeypatch.setattr(model, 'generate', generate)
    monkeypatch.setattr(run_baseline, 'load_model', lambda *args: (model, tok))
    # The actual prefill and held-out decoding paths run on the tiny model.
    source = tmp_path / 'split.json'
    source.write_text(json.dumps({'context': 'The observatory is Silver Ridge. '*3,
        'test_queries': ['\nQuestion: Name?\nAnswer:', '\nQuestion: Year?\nAnswer:'],
        'metadata': {'references': [['Silver Ridge'], ['1998']], 'metric_kind': 'any',
            'task': 'custom', 'record_id': 'one', 'max_length': 512, 'generation_tokens': 2}}))
    output = tmp_path / 'result.json'
    monkeypatch.setattr(sys, 'argv', ['baseline', '--method', 'summary', '--model', 'tiny',
        '--split', str(source), '--budget', '16', '--device', 'cpu', '--output', str(output)])
    run_baseline.main()
    result = json.loads(output.read_text())
    assert len(generated) == 1
    assert result['queries'] == 2 and len(result['rows']) == 2
    assert result['summary']['summary_tokens'] <= 16
    assert 'Name?' not in tok.messages[0][0][0]['content']
    assert 'Year?' not in tok.messages[0][0][0]['content']
