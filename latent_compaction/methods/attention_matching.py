"""Attention Matching: RMS key selection, bounded NNLS biases, and LSQ values."""
import math
from pathlib import Path
import sys
import torch
from ..evaluation import full_cache
from ..memory import selected_memory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party/attention_matching'))


@torch.no_grad()
def construct(model, tokenizer, document, budget):
    from compaction.algorithms.highest_attention_keys import HighestAttentionKeysCompaction
    exact = full_cache(model, tokenizer, document)
    length = exact.get_seq_length()
    if not 0 < budget < length:
        raise ValueError('budget must be positive and smaller than context length')
    fitter = HighestAttentionKeysCompaction(score_method='rms', nnls_iters=2,
        nnls_lower_bound=math.exp(-3), nnls_upper_bound=math.exp(3), c2_method='lsq')
    device = next(model.parameters()).device
    repeat = tokenizer(' Repeat the previous context. ' + document,
        return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    reference = exact.fork(capture_queries=True)
    for chunk in repeat.split(256, dim=-1):
        model.model(input_ids=chunk, past_key_values=reference, use_cache=True)
    banks = {i: torch.cat(chunks, dim=-2).to(device) for i, chunks in reference.queries.items()}
    del reference
    memories = {}
    for i, layer in enumerate(exact.layers):
        heads = layer.keys.shape[1]
        groups = model.config.num_attention_heads // heads
        keys, values, biases = [], [], []
        for head in range(heads):
            queries = banks[i][0, head*groups:(head+1)*groups].reshape(-1, layer.keys.shape[-1])
            k, bias, v, _ = fitter.compute_compacted_cache(layer.keys[0, head], layer.values[0, head], queries, budget)
            keys.append(k); values.append(v); biases.append(bias.flatten())
        memories[i] = selected_memory(model, i, torch.stack(keys)[None],
            torch.stack(values)[None], bias=torch.stack(biases))
    return exact, memories, length
