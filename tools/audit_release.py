#!/usr/bin/env python3
"""Check release files for private paths, credential patterns and generated artifacts."""
import argparse
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    rules = {
        'private home path': r'/(?:Users|home)/[A-Za-z0-9_.-]+/',
        'private deployment path': r'/(?:root|private)/(?:latent|research|recovery)',
        'private IPv4 address': r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b',
        'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'credential': r'\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b',
        'assigned secret': r'(?i)(?:api_key|password|access_token)\s*[=:]\s*["\x27][^"\x27]{8,}["\x27]',
    }
    findings = []
    count = 0
    for path in args.root.rglob('*'):
        if not path.is_file() or any(x in path.parts for x in ('.git', '__pycache__', '.pytest_cache', '.venv')) or any(x.endswith('.egg-info') for x in path.parts):
            continue
        relative = str(path.relative_to(args.root))
        if path.suffix in ('.pt', '.pth', '.safetensors', '.bin', '.log') or path.name.startswith('.env'):
            findings.append({'file': relative, 'rule': 'generated or secret-bearing artifact'})
        if path.suffix == '.png' and relative.startswith('assets/') and path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):
            count += 1
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            findings.append({'file': relative, 'rule': 'unexpected binary file'})
            continue
        count += 1
        for rule, pattern in rules.items():
            if re.search(pattern, text):
                findings.append({'file': relative, 'rule': rule})
    print(json.dumps({'files_scanned': count, 'findings': findings}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == '__main__':
    main()
