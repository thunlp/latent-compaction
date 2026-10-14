"""Average documents within tasks, then equally weight tasks; never pool queries."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from latent_compaction.evaluation import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cells = defaultdict(list)
    for path in a.results.glob('r*/*/result.json'):
        row = json.loads(path.read_text())
        cells[(path.parent.parent.name, row.get('length'), row['task'])].append(row)
    if not cells:
        p.error('no completed result.json files found')
    tasks = [{'ratio': ratio, 'length': length, 'task': task, 'documents': len(rows),
              'score': sum(r['score'] for r in rows) / len(rows),
              'full_cache_score': sum(r['full_cache_score'] for r in rows) / len(rows)}
             for (ratio, length, task), rows in sorted(cells.items())]
    overall = []
    for ratio, length in sorted({(t['ratio'], t['length']) for t in tasks}):
        selected = [t for t in tasks if t['ratio'] == ratio and t['length'] == length]
        overall.append({'ratio': ratio, 'length': length, 'tasks': len(selected),
                        'score': sum(t['score'] for t in selected) / len(selected)})
    write_json(a.output, {'tasks': tasks, 'macro_average': overall,
                         'note': 'Partial runs report completed documents only; verify cohort completeness before comparing to the paper.'})
    print(json.dumps(overall, indent=2))


if __name__ == '__main__':
    main()
