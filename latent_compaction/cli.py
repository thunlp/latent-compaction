"""Fit, generate, and score held-out queries with portable, explicit settings."""
from __future__ import annotations
import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import platform
import torch
import transformers
from .models import generate_from_memory, load_model, resolve_model_name
from .methods.latent import FitConfig, fit_document, load_memory, save_memory
from .evaluation import content_hash, full_cache, generate_exact, metric, validate_split, write_json


def add_model_options(parser):
    parser.add_argument('--model', required=True, help='model alias, Hub ID or local checkpoint')
    parser.add_argument('--device', default='cuda', help='cuda, cuda:N, cpu or mps')


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog='python -m latent_compaction.cli')
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit', help='train a reusable document cache')
    add_model_options(fit)
    inputs = fit.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--document', type=Path, help='UTF-8 text')
    inputs.add_argument('--split', type=Path, help='benchmark JSON; only context is used for training')
    fit.add_argument('--output', required=True, type=Path)
    fit.add_argument('--config', type=Path, help='JSON FitConfig')
    fit.add_argument('--budget', type=int)
    fit.add_argument('--soft-slots', type=int)
    fit.add_argument('--ratio', type=int, help='nominal max_length / budget, from split metadata')
    fit.add_argument('--stage1-steps', type=int)
    fit.add_argument('--stage2-steps', type=int)
    fit.add_argument('--seed', type=int)
    generate = sub.add_parser('generate', help='answer from a saved cache')
    add_model_options(generate)
    generate.add_argument('--cache', type=Path, required=True)
    generate.add_argument('--prompt', required=True)
    generate.add_argument('--max-new-tokens', type=int, default=128)
    evaluate = sub.add_parser('evaluate', help='score held-out queries and the full-cache reference')
    add_model_options(evaluate)
    evaluate.add_argument('--cache', type=Path, required=True)
    evaluate.add_argument('--split', type=Path, required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--max-new-tokens', type=int)
    args = parser.parse_args(argv)
    if args.command == 'fit':
        values = json.loads(args.config.read_text()) if args.config else {}
        unknown = set(values) - {f.name for f in fields(FitConfig)}
        if unknown:
            parser.error('unknown config fields: ' + ', '.join(sorted(unknown)))
        for name in ('budget', 'soft_slots', 'stage1_steps', 'stage2_steps', 'seed'):
            if getattr(args, name) is not None:
                values[name] = getattr(args, name)
        split = validate_split(json.loads(args.split.read_text())) if args.split else None
        if args.ratio is not None:
            if not split or args.ratio < 1:
                parser.error('--ratio requires a split and a positive ratio')
            n = split['metadata']['max_length']
            if split['metadata'].get('metric_kind') == 'longbench':
                n = split['metadata']['context_tokens']
                values['budget'] = max(4, (n // args.ratio) // 4 * 4)
            else:
                values['budget'] = n // args.ratio
            values['soft_slots'] = values['budget'] // 4
        if not all(name in values for name in ('budget', 'soft_slots')):
            parser.error('specify --budget and --soft-slots, --ratio, or a config containing both')
        config = FitConfig(**values)
        document = split['context'] if split else args.document.read_text(encoding='utf-8')
    model, tokenizer = load_model(args.model, device=args.device)
    model_id = resolve_model_name(args.model)
    if args.command == 'fit':
        memories, tokens = fit_document(model, tokenizer, document, config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_memory(args.output, memories, tokens, model_id=model_id)
        write_json(args.output.with_suffix('.metadata.json'), {
            'protocol': 'paper-two-stage-1000-default', 'fit_config': asdict(config),
            'input_sha256': content_hash(args.split or args.document),
            'document_tokens': tokens, 'model_id': model_id,
            'model_config': model.config.to_dict(),
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'python': platform.python_version(), 'evaluation_queries_used_for_training': False,
        })
        print('Saved cache to', args.output)
        return
    memories, tokens = load_memory(args.cache, args.device, expected_model_id=model_id)
    if args.command == 'generate':
        if args.max_new_tokens < 1:
            parser.error('max-new-tokens must be positive')
        print(generate_from_memory(model, tokenizer, memories, tokens, args.prompt, max_new_tokens=args.max_new_tokens))
        return
    data = validate_split(json.loads(args.split.read_text()))
    metadata = data['metadata']
    receipt = args.cache.with_suffix('.metadata.json')
    if not receipt.exists() or json.loads(receipt.read_text())['input_sha256'] != content_hash(args.split):
        parser.error('cache provenance does not match this split; fit using --split')
    cap = args.max_new_tokens or metadata.get('generation_tokens', 128)
    if cap < 1:
        parser.error('generation cap must be positive')
    exact = full_cache(model, tokenizer, data['context'])
    rows = []
    for prompt, references in zip(data['test_queries'], metadata['references']):
        prediction = generate_from_memory(model, tokenizer, memories, tokens, prompt, max_new_tokens=cap)
        reference = generate_exact(model, tokenizer, exact, prompt, cap)
        kind = metadata.get('metric_kind', 'coverage')
        rows.append({'prediction': prediction, 'full_cache_prediction': reference,
                     'score': metric(prediction, references, kind, metadata),
                     'full_cache_score': metric(reference, references, kind, metadata)})
    write_json(args.output, {'task': metadata.get('task', 'custom'), 'length': metadata.get('max_length'), 'record_id': metadata.get('record_id'),
        'split_sha256': content_hash(args.split), 'cache_sha256': content_hash(args.cache),
        'model_id': model_id, 'generation_tokens': cap, 'metric_kind': kind,
        'score': sum(r['score'] for r in rows) / len(rows),
        'full_cache_score': sum(r['full_cache_score'] for r in rows) / len(rows),
        'queries': len(rows), 'rows': rows})
    print('Saved evaluation to', args.output)


if __name__ == '__main__':
    main()
