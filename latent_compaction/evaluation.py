"""Held-out metrics and native full-cache reference generation."""
from __future__ import annotations
import json
import re
import hashlib
from pathlib import Path
import torch
from .models import DocumentCache


def validate_split(data):
    if not isinstance(data.get('context'), str) or not data['context'].strip():
        raise ValueError('split requires nonempty context')
    queries = data.get('test_queries', [])
    references = data.get('metadata', {}).get('references', [])
    if not queries or len(queries) != len(references):
        raise ValueError('test_queries and metadata.references must have equal nonzero length')
    if any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError('each query must be a nonempty string')
    if any(not r or any(not isinstance(a, str) or not a for a in r) for r in references):
        raise ValueError('each question requires at least one nonempty reference')
    return data


def metric(prediction, references, kind='coverage', metadata=None):
    if kind == 'phonebook':
        match = re.search(r'(?<!\d)\d{10}(?!\d)', prediction)
        return float(bool(match) and match.group() == references[0])
    if kind == 'niah':
        normalize = lambda s: re.sub(r'[^a-z0-9-]', '', s.lower())
        return float(any(normalize(a) in normalize(prediction) for a in references))
    if kind == 'longbench':
        import sys
        root = Path(__file__).resolve().parents[1] / 'third_party' / 'longbench'
        sys.path.insert(0, str(root))
        from eval import dataset2metric
        task = metadata['task']
        if task in ('trec', 'triviaqa', 'samsum'):
            prediction = prediction.lstrip('\n').split('\n')[0]
        return max(dataset2metric[task](prediction, a, all_classes=metadata.get('all_classes', [])) for a in references)
    if kind == 'exact':
        return float(prediction.strip() in references)
    if kind not in ('coverage', 'all', 'any'):
        raise ValueError('unknown metric: ' + kind)
    hits = [float(a.lower() in prediction.lower()) for a in references]
    return max(hits) if kind == 'any' else sum(hits) / len(hits)


@torch.no_grad()
def full_cache(model, tokenizer, document, *, capture_queries=False):
    ids = tokenizer(document, return_tensors='pt', add_special_tokens=False).input_ids.to(next(model.parameters()).device)
    cache = DocumentCache(model.config)
    cache.capture_queries = capture_queries
    model.model(input_ids=ids, past_key_values=cache, use_cache=True)
    return cache


@torch.no_grad()
def generate_exact(model, tokenizer, cache, prompt, max_new_tokens=128):
    ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(next(model.parameters()).device)
    current = cache.fork()
    logits = model(input_ids=ids, past_key_values=current, use_cache=True).logits[:, -1]
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    generated = []
    for _ in range(max_new_tokens):
        token = logits.argmax(-1, keepdim=True)
        if token.item() in eos:
            break
        generated.append(int(token.item()))
        logits = model(input_ids=token, past_key_values=current, use_cache=True).logits[:, -1]
    return tokenizer.decode(generated, skip_special_tokens=True)


def content_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)
