"""PyramidKV: pooled tail-attention selection with exact-total layer allocation."""
from ..eviction import compact_by_tail_attention, layer_budgets, WINDOW


def construct(model, tokenizer, document, budget):
    if budget < 1:
        raise ValueError('budget must be positive')
    window = min(WINDOW, budget)
    counts = lambda length: layer_budgets('pyramidkv', budget,
        model.config.num_hidden_layers, window=window, max_context=length)
    return compact_by_tail_attention(model, tokenizer, document, counts, window=window)
