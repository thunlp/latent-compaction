"""Protocol invariants: masking, precision, bias schedules, metrics and source leakage."""
import json
from pathlib import Path
import sys
import tempfile
import torch
from latent_compaction.memory import CompactedMemory
from latent_compaction.evaluation import metric, validate_split
from latent_compaction.methods.latent import FitConfig
from latent_compaction.prompts import SOURCE_PROMPTS


def test_bias_schedule_and_masked_normalizer():
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn_like(k)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    soft = torch.randn(1, 2, 2, 4)
    m = CompactedMemory(k, v, mask, soft, soft, query_heads=4, scaling=.5)
    assert m.soft_bias.requires_grad
    assert not m.preserver_keys.requires_grad and not m.preserver_bias.requires_grad
    q = torch.randn(1, 4, 3, 4)
    output, log_z = m(q)
    keys = torch.cat((k[0], soft[0]), dim=1).repeat_interleave(2, 0)
    values = torch.cat((v[0], soft[0]), dim=1).repeat_interleave(2, 0)
    valid = torch.cat((mask, torch.ones(2, 2, dtype=torch.bool)), dim=1).repeat_interleave(2, 0)
    scores = torch.einsum('bhqd,hnd->bhqn', q, keys)*.5
    scores = scores.masked_fill(~valid[None, :, None, :], float('-inf'))
    torch.testing.assert_close(log_z, scores.logsumexp(-1))
    torch.testing.assert_close(output, torch.einsum('bhqn,hnd->bhqd', scores.softmax(-1), values))
    m.stage_two()
    assert all(p.requires_grad for p in m.parameters())


def test_paper_schedule():
    config = FitConfig(budget=1024, soft_slots=256)
    assert config.stage1_steps == config.stage2_steps == 1000
    assert config.stage1_queries == 4096
    assert config.stage1_batch_size == 128 and config.stage2_batch_size == 4


def test_metrics_and_validation():
    assert metric('The number is 0123456789.', ['0123456789'], 'phonebook') == 1
    assert metric('Wrong 9999999999 then 0123456789', ['0123456789'], 'phonebook') == 0
    assert metric('a', ['a', 'b'], 'all') == .5
    assert metric('b', ['a', 'b'], 'any') == 1
    assert metric('A!B-1', ['ab-1'], 'niah') == 1
    try:
        validate_split({'context': 'abc', 'test_queries': ['q'], 'metadata': {'references': []}})
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched references accepted')


def test_deterministic_phonebook_and_source_bank():
    from prepare import phonebook
    from types import SimpleNamespace
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()
    with tempfile.TemporaryDirectory() as directory:
        args = SimpleNamespace(output=Path(directory), lengths=[512], samples=1)
        rows = []
        phonebook(args, Tokenizer(), rows)
        first = (args.output / rows[0]['split']).read_bytes()
        phonebook(args, Tokenizer(), [])
        assert first == (args.output / rows[0]['split']).read_bytes()
        data = json.loads(first)
        assert data['train_queries'] == list(SOURCE_PROMPTS)
        assert len(data['test_queries']) == 20
        assert not set(data['test_queries']) & set(data['train_queries'])
