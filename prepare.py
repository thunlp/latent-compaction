#!/usr/bin/env python3
"""Deterministic phonebook, official RULER, and LongBench-v1 preparation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import string
import subprocess
import sys
import zipfile
from transformers import AutoTokenizer
from latent_compaction.prompts import SOURCE_PROMPTS
from latent_compaction.models import resolve_model_name
from latent_compaction.evaluation import write_json

ROOT = Path(__file__).resolve().parent
TASKS = ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique',
         'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum',
         'passage_count', 'passage_retrieval_en', 'lcc', 'repobench-p']


def save(args, tok, records, task, length, index, context, queries, references, kind, extra=None):
    name = f'{task}-{length}-{index:05d}'
    metadata = {'task': task, 'record_id': name, 'max_length': length,
                'context_tokens': len(tok.encode(context, add_special_tokens=False)),
                'references': references, 'answers': [r[0] for r in references],
                'metric_kind': kind, 'generation_tokens': 128,
                'evaluation_query_used_for_compression': False, **(extra or {})}
    data = {'format_version': 1, 'context': context, 'train_queries': list(SOURCE_PROMPTS),
            'validation_queries': [], 'test_queries': queries, 'metadata': metadata}
    path = args.output / 'splits' / (name + '.json')
    if path.exists() and json.loads(path.read_text()) != data:
        raise ValueError(f'Refusing to overwrite a different split: {path}')
    write_json(path, data)
    records.append({'id': name, 'task': task, 'length': length, 'index': index,
                    'split': str(path.relative_to(args.output)),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'context_tokens': metadata['context_tokens'],
                    'seed': json.loads((ROOT / 'configs/document-seeds.json').read_text()).get(name, 401 + len(records))})


def phonebook(args, tok, records):
    for length in args.lengths:
        for i in range(args.samples):
            seed = 918000 + length * 100 + i
            rng = random.Random(seed)
            entries, names = [], set()
            while len(entries) < length // 8:
                name = ''.join(rng.choices(string.ascii_uppercase, k=10))
                number = ''.join(rng.choices(string.digits, k=10))
                if name in names:
                    continue
                names.add(name)
                entries.append({'name': name, 'number': number})
            lines = [f'The phone number of {e["name"]} is {e["number"]}.' for e in entries]
            lo, hi = 0, len(lines)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(tok.encode('\n'.join(lines[:mid]), add_special_tokens=False)) <= length - 128:
                    lo = mid
                else:
                    hi = mid - 1
            if lo < 20:
                raise ValueError('context too short for 20 phonebook lookups')
            selected = random.Random(seed + 1).sample(entries[:lo], 20)
            queries = [f'\nQuestion: What is the phone number of {e["name"]}?\nAnswer:' for e in selected]
            save(args, tok, records, 'phonebook', length, i, '\n'.join(lines[:lo]), queries,
                 [[e['number']] for e in selected], 'phonebook',
                 {'seed': seed, 'entries_count': lo, 'evaluation_entries': selected})


def ruler(args, tok, records):
    import yaml
    if args.upstream is None:
        raise ValueError('--upstream must point to a pinned NVIDIA/RULER checkout')
    up = args.upstream / 'scripts'
    config = yaml.safe_load((up / 'synthetic.yaml').read_text())
    spec = importlib.util.spec_from_file_location('ruler_constants', up / 'data/synthetic/constants.py')
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    for task, entry in config.items():
        if args.tasks and task not in args.tasks:
            continue
        base = constants.TASKS[entry['task']]
        for length in args.lengths:
            dest = args.output / 'raw' / str(length)
            raw = dest / task / 'validation.jsonl'
            if not raw.exists():
                command = [sys.executable, str(up / 'data/synthetic' / (entry['task'] + '.py')),
                    '--save_dir', str(dest), '--save_name', task, '--subset', 'validation',
                    '--tokenizer_path', args.model, '--tokenizer_type', 'hf',
                    '--max_seq_length', str(length), '--tokens_to_generate', str(base['tokens_to_generate']),
                    '--num_samples', str(args.samples), '--random_seed', '42',
                    '--template', base['template'] + base['answer_prefix']]
                for key, value in entry['args'].items():
                    command.extend(['--' + key, str(value)])
                subprocess.run(command, check=True)
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            if len(rows) != args.samples:
                raise ValueError('existing raw RULER sample count differs')
            for i, row in enumerate(rows):
                text = row['input'] + row.get('answer_prefix', '')
                marker = '\nWhat ' if entry['task'] == 'niah' else (
                    '\n\nAnswer the question based on the given documents.' if entry['task'] == 'qa' else '\nQuestion:')
                where = text.rfind(marker)
                if where <= 0:
                    raise ValueError('unsupported RULER question boundary: ' + task)
                save(args, tok, records, task, length, i, text[:where], [text[where:]], [row['outputs']],
                     'any' if entry['task'] == 'qa' else 'all',
                     {'source': 'NVIDIA/RULER', 'generation_tokens': base['tokens_to_generate'],
                      'raw_record_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()})


def longbench(args, tok, records):
    from huggingface_hub import hf_hub_download
    source = args.data_zip or Path(hf_hub_download('THUDM/LongBench', 'data.zip', repo_type='dataset',
                                      revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672'))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64':
        raise ValueError('LongBench data.zip checksum differs from the experiment snapshot')
    templates = json.loads((ROOT / 'third_party/longbench/config/dataset2prompt.json').read_text())
    caps = json.loads((ROOT / 'third_party/longbench/config/dataset2maxlen.json').read_text())
    with zipfile.ZipFile(source) as archive:
        for ti, task in enumerate(args.tasks or TASKS):
            names = [n for n in archive.namelist() if n.endswith('/' + task + '.jsonl') or n == task + '.jsonl']
            if len(names) != 1:
                raise ValueError('cannot identify LongBench task ' + task)
            rows = [json.loads(line) for line in archive.read(names[0]).decode().splitlines()]
            indices = sorted(random.Random(42).sample(range(len(rows)), args.samples))
            for i in indices:
                row = rows[i]
                pre, post = templates[task].split('{context}')
                query = post.format(**row)
                full = pre + row['context']
                ids = tok.encode(full, add_special_tokens=False)
                allow = 16384 - len(tok.encode(query, add_special_tokens=False))
                if allow <= 16:
                    raise ValueError('query exceeds input cap')
                context = full
                if len(ids) > allow:
                    while allow > 16:
                        half = allow // 2
                        context = tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-(allow-half):], skip_special_tokens=True)
                        if (len(tok.encode(context, add_special_tokens=False)) + len(tok.encode(query, add_special_tokens=False)) <= 16384
                            and len(tok.encode(context + query, add_special_tokens=False)) <= 16384):
                            break
                        allow -= 8
                    else:
                        raise ValueError('unable to fit LongBench context')
                save(args, tok, records, task, 16384, i, context, [query], [row['answers']], 'longbench',
                     {'all_classes': row.get('all_classes', []), 'generation_tokens': caps[task],
                      'source_index': i, 'middle_truncated': context != full,
                      'raw_record_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()})
                records[-1]['seed'] = 601000 + TASKS.index(task)*10000 + i


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['phonebook', 'ruler', 'longbench'])
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--lengths', nargs='+', type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--data-zip', type=Path)
    args = parser.parse_args()
    if args.samples < 1 or any(n < 512 for n in args.lengths):
        parser.error('samples must be positive and context lengths at least 512')
    args.model = resolve_model_name(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    records = []
    {'phonebook': phonebook, 'ruler': ruler, 'longbench': longbench}[args.benchmark](args, tok, records)
    write_json(args.output / 'manifest.json', {'benchmark': args.benchmark, 'model': args.model,
        'samples_per_cell': args.samples, 'documents': records, 'source_prompts': len(SOURCE_PROMPTS)})
    print('Prepared', len(records), 'documents in', args.output)


if __name__ == '__main__':
    main()
