#!/usr/bin/env python3
"""Run any method on one document or a dataset manifest."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from latent_compaction.evaluation import content_hash, validate_split, write_json

ROOT = Path(__file__).resolve().parent
METHODS = ('latent', 'full', 'kvzip', 'attention_matching', 'snapkv', 'pyramidkv', 'summary')


def run_cell(args, split, budget, seed, output, ratio=None):
    output.mkdir(parents=True, exist_ok=True)
    cache, result = output / 'cache.pt', output / 'result.json'
    receipt = {'method': args.method, 'model': args.model, 'device': args.device,
        'budget': budget, 'ratio': ratio, 'seed': seed, 'split_sha256': content_hash(split),
        'config_sha256': content_hash(args.config), 'runner_sha256': content_hash(__file__)}
    settings = output / 'settings.json'
    if settings.exists() and json.loads(settings.read_text()) != receipt:
        raise ValueError('output settings differ; use a new output directory')
    write_json(settings, receipt)
    if result.exists():
        finished = json.loads(result.read_text())
        if finished['split_sha256'] != receipt['split_sha256']:
            raise ValueError('finished result has a different input')
        if args.method == 'latent' and finished['cache_sha256'] != content_hash(cache):
            raise ValueError('finished cache checksum changed')
        print('Reuse', output, flush=True)
        return
    common = ['--model', args.model, '--device', args.device, '--split', str(split)]
    if args.method == 'latent':
        subprocess.run([sys.executable, '-m', 'latent_compaction.cli', 'fit', *common,
            '--config', str(args.config), '--budget', str(budget), '--soft-slots', str(budget // 4),
            '--seed', str(seed), '--output', str(cache)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, '-m', 'latent_compaction.cli', 'evaluate', *common,
            '--cache', str(cache), '--output', str(result)], check=True, cwd=ROOT)
    else:
        subprocess.run([sys.executable, '-m', 'latent_compaction.runner', *common,
            '--method', args.method, '--budget', str(budget), '--seed', str(seed),
            '--output', str(result)], check=True, cwd=ROOT)


def budget_for(metadata, ratio):
    if metadata.get('metric_kind') == 'longbench':
        return max(4, (metadata['context_tokens'] // ratio) // 4 * 4)
    return metadata['max_length'] // ratio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, default='latent')
    p.add_argument('--model', required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--split', type=Path, help='one document JSON')
    inputs.add_argument('--manifest', type=Path, help='prepared dataset manifest')
    inputs.add_argument('--cache', type=Path, help='answer a new question using a fitted cache')
    p.add_argument('--prompt', help='question suffix for --cache')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--lengths', nargs='+', type=int, help='select nominal context lengths from a manifest')
    p.add_argument('--budget', type=int, help='explicit single-document budget')
    p.add_argument('--ratios', nargs='+', type=int, default=[16, 32])
    p.add_argument('--config', type=Path, default=ROOT/'configs/paper.json')
    p.add_argument('--seed', type=int, default=401, help='single-document seed')
    p.add_argument('--output', type=Path, help='directory for training/evaluation outputs')
    p.add_argument('--device', default='cuda')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    args = p.parse_args()
    for name in ('split', 'manifest', 'cache', 'output', 'config'):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    if Path(args.model).is_dir():
        args.model = str(Path(args.model).resolve())
    if args.cache:
        if args.method != 'latent' or not args.prompt:
            p.error('--cache requires --method latent and --prompt')
        from latent_compaction.cli import main as cache_cli
        cache_cli(['generate', '--model', args.model, '--device', args.device,
                   '--cache', str(args.cache), '--prompt', args.prompt,
                   '--max-new-tokens', str(args.max_new_tokens)])
        return
    if args.output is None:
        p.error('--output is required for training/evaluation')
    if not 0 <= args.shard < args.num_shards or any(r < 1 for r in args.ratios):
        p.error('invalid shard or ratio')
    if args.budget is not None and (args.manifest or args.budget < 4):
        p.error('--budget must be at least four and used with --split')
    if args.split:
        data = validate_split(json.loads(args.split.read_text()))
        if args.budget is not None:
            run_cell(args, args.split, args.budget, args.seed, args.output)
        else:
            for ratio in args.ratios:
                run_cell(args, args.split, budget_for(data['metadata'], ratio), args.seed,
                         args.output / ('r' + str(ratio)), ratio)
        return
    manifest = json.loads(args.manifest.read_text())
    for index, row in enumerate(manifest['documents']):
        if index % args.num_shards != args.shard:
            continue
        split = args.manifest.parent / row['split']
        if content_hash(split) != row['sha256']:
            raise ValueError('split checksum changed: ' + row['id'])
        metadata = validate_split(json.loads(split.read_text()))['metadata']
        if args.lengths and metadata['max_length'] not in args.lengths:
            continue
        for ratio in args.ratios:
            run_cell(args, split, budget_for(metadata, ratio), row['seed'],
                     args.output / ('r' + str(ratio)) / row['id'], ratio)


if __name__ == '__main__':
    main()
