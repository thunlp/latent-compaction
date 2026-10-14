"""SnapKV: pooled tail-attention selection with a uniform layer budget."""
from ..eviction import compact_by_tail_attention, WINDOW


def construct(model, tokenizer, document, budget):
    if budget < 1:
        raise ValueError('budget must be positive')
    counts = [budget] * model.config.num_hidden_layers
    return compact_by_tail_attention(model, tokenizer, document, counts, window=min(WINDOW, budget))
