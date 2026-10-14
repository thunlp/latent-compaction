"""Qwen3/GQA adaptations of the original SnapKV and PyramidKV algorithms.

Upstream MIT license notices are retained under third_party/licenses/.
Pyramid's integer step truncation is replaced by exact-total linear rounding.
"""
import math
from fractions import Fraction
import torch
from torch.nn import functional as F

WINDOW=32
KERNEL=5
BETA=20

def layer_budgets(method, budget=128, layers=28, window=WINDOW, max_context=None):
    window=min(window,budget)
    if method=='snapkv':return [budget]*layers
    if method!='pyramidkv':raise ValueError(method)
    assert layers>1 and budget>=window
    minimum=(budget-window)//BETA
    maximum=2*(budget-window)-minimum
    counts=[window+round(Fraction(maximum*(layers-1)-i*(maximum-minimum),layers-1)) for i in range(layers)]
    if max_context is not None:
        assert budget<=max_context
        counts=[min(c,max_context) for c in counts]
        deficit=layers*budget-sum(counts)
        # Capped water filling; earlier layers break ties deterministically.
        while deficit:
            available=[i for i,c in enumerate(counts) if c<max_context]
            assert available
            share=max(1,deficit//len(available))
            for i in available:
                add=min(share,max_context-counts[i],deficit)
                counts[i]+=add;deficit-=add
                if not deficit:break
    assert sum(counts)==layers*budget and min(counts)>=window
    assert max_context is None or max(counts)<=max_context
    return counts

@torch.no_grad()
def select_indices(keys,queries,budget,window=WINDOW,kernel=KERNEL):
    """keys [1,Hkv,N,D], final W post-RoPE queries [1,Hq,W,D]."""
    batch,heads,n,d=keys.shape
    assert batch==1 and queries.shape[-2]==window and window<=budget<=n
    if budget == window:
        return torch.arange(n-window,n,device=keys.device)[None,None,:].expand(batch,heads,-1)
    groups=queries.shape[1]//heads
    repeated=keys.repeat_interleave(groups,dim=1)
    logits=(queries@repeated.transpose(-1,-2))/math.sqrt(d)
    mask=torch.full((window,window),torch.finfo(logits.dtype).min,device=keys.device)
    positions=torch.arange(window,device=keys.device)
    mask.masked_fill_(positions[None,:]<=positions[:,None],0)
    logits[...,-window:]+=mask
    attention=logits.softmax(-1,dtype=torch.float32).to(queries.dtype)
    scores=attention[...,:-window].sum(-2)
    scores=F.avg_pool1d(scores,kernel_size=kernel,padding=kernel//2,stride=1)
    # Keep a shared GQA cache, not a doubled query-head KV cache.
    scores=scores.reshape(batch,heads,groups,n-window).mean(2)
    selected=scores.topk(budget-window,dim=-1).indices
    recent=torch.arange(n-window,n,device=keys.device)[None,None,:].expand(batch,heads,-1)
    return torch.cat([selected,recent],-1)

@torch.no_grad()
def compact_by_tail_attention(model, tokenizer, document, counts, *, window=WINDOW):
    """Shared document-only observation-window scoring for SnapKV/PyramidKV."""
    from .evaluation import full_cache
    from .memory import selected_memory
    exact = full_cache(model, tokenizer, document, capture_queries=True)
    length = exact.get_seq_length()
    counts = counts(length) if callable(counts) else counts
    memories = {}
    for i, layer in enumerate(exact.layers):
        queries = torch.cat(exact.queries[i], dim=-2).to(layer.keys.device)[..., -window:, :]
        if not window <= counts[i] <= length:
            raise ValueError('layer budget must fit the document and observation window')
        indices = select_indices(layer.keys, queries, counts[i], window=window)
        gather = indices[..., None].expand(-1, -1, -1, layer.keys.shape[-1])
        memories[i] = selected_memory(model, i, layer.keys.gather(2, gather), layer.values.gather(2, gather))
    exact.capture_queries = False
    exact.queries.clear()
    return exact, memories, length
