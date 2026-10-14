"""KVZip+: global reconstruction-based KV selection at a fixed average budget."""
import torch
from ..models import DocumentCache
from ..memory import selected_memory
from ..selection import capture_kvzip_scores, select_preservers, gather_preservers


@torch.no_grad()
def construct(model, tokenizer, document, budget):
    device = next(model.parameters()).device
    ids = tokenizer(document, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    length = ids.shape[-1]
    if not 0 < budget < length:
        raise ValueError('budget must be positive and smaller than context length')
    exact = DocumentCache(model.config)
    scores = capture_kvzip_scores(model, ids, exact, budget / length)
    masks = select_preservers(scores, budget)
    memories = {}
    for i, layer in enumerate(exact.layers):
        keys, values, mask = gather_preservers(layer.keys, layer.values, masks[i, 0])
        memories[i] = selected_memory(model, i, keys, values, mask)
    return exact, memories, length
