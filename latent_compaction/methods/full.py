"""Uncompressed frozen-reader reference."""
from ..evaluation import full_cache


def construct(model, tokenizer, document, budget):
    exact = full_cache(model, tokenizer, document)
    return exact, None, exact.get_seq_length()
