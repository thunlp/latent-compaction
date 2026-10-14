"""Shared held-out evaluation for all baseline methods."""
import argparse
import importlib
import json
from pathlib import Path
import torch
from .models import load_model, resolve_model_name, generate_from_memory
from .methods.summary import summarize_document
from .evaluation import full_cache, generate_exact, validate_split, metric, content_hash, write_json

BASELINES = ('full', 'kvzip', 'attention_matching', 'snapkv', 'pyramidkv', 'summary')


def construct(model, tokenizer, document, method, budget):
    if method not in BASELINES or method == 'summary':
        raise ValueError('unknown KV baseline: ' + method)
    module = importlib.import_module('latent_compaction.methods.' + method)
    return module.construct(model, tokenizer, document, budget)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['full', 'kvzip', 'attention_matching', 'snapkv', 'pyramidkv', 'summary'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    data = validate_split(json.loads(args.split.read_text()))
    torch.manual_seed(args.seed)
    model, tok = load_model(args.model, args.device)
    summary = None
    if args.method == 'summary':
        summary = summarize_document(model, tok, data['context'], args.budget)
        summary_cache = full_cache(model, tok, summary['summary'])
        exact = full_cache(model, tok, data['context'])
        memories = None
    else:
        exact, memories, length = construct(model, tok, data['context'], args.method, args.budget)
    meta = data['metadata'];cap = meta.get('generation_tokens', 128)
    rows = []
    for q, refs in zip(data['test_queries'], meta['references']):
        reference = generate_exact(model, tok, exact, q, cap)
        if summary is not None:
            prediction = generate_exact(model, tok, summary_cache, q, cap)
        elif memories is None:
            prediction = reference
        else:
            prediction = generate_from_memory(model, tok, memories, length, q, max_new_tokens=cap)
        rows.append({'prediction': prediction, 'full_cache_prediction': reference,
                     'score': metric(prediction, refs, meta['metric_kind'], meta),
                     'full_cache_score': metric(reference, refs, meta['metric_kind'], meta)})
    write_json(args.output, {'method': args.method, 'model_id': resolve_model_name(args.model),
        'task': meta['task'], 'length': meta['max_length'], 'record_id': meta['record_id'], 'budget': args.budget,
        'split_sha256': content_hash(args.split), 'seed': args.seed, 'summary': summary, 'queries': len(rows), 'rows': rows,
        'score': sum(r['score'] for r in rows)/len(rows),
        'full_cache_score': sum(r['full_cache_score'] for r in rows)/len(rows)})


if __name__ == '__main__':
    main()
