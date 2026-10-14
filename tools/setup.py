#!/usr/bin/env python3
"""Install the pinned release dependencies, with an explicit KVPress metadata patch."""
import argparse
import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.check_only:
        in_venv = sys.prefix != sys.base_prefix
        in_conda = os.environ.get('CONDA_PREFIX') == sys.prefix
        if not (in_venv or in_conda):
            parser.error('Activate a virtual environment first: python -m venv .venv')
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps',
                            '--dest', directory, 'kvpress==0.5.4'], check=True)
            source = next(Path(directory).glob('kvpress-*.whl'))
            with zipfile.ZipFile(source) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
            metadata = next(name for name in files if name.endswith('/METADATA'))
            original = files[metadata].decode()
            old = next(line for line in original.splitlines()
                       if line.startswith('Requires-Dist: transformers'))
            # No KVPress code is changed. Scope compatibility to the tested version.
            patched_metadata = original.replace(old, 'Requires-Dist: transformers==5.6.0')
            fire_requirement = next(line for line in original.splitlines()
                                    if line.startswith('Requires-Dist: fire'))
            patched_metadata = patched_metadata.replace(fire_requirement, 'Requires-Dist: fire>=0.7.1,<0.8')
            files[metadata] = patched_metadata.encode()
            record = next(name for name in files if name.endswith('/RECORD'))
            rows = []
            for name, data in files.items():
                if name == record:
                    continue
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                rows.append([name, 'sha256=' + digest, str(len(data))])
            rows.append([record, '', ''])
            buf = io.StringIO()
            csv.writer(buf, lineterminator='\n').writerows(rows)
            files[record] = buf.getvalue().encode()
            patched = Path(directory) / 'patched' / source.name
            patched.parent.mkdir()
            with zipfile.ZipFile(patched, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name, data in files.items():
                    archive.writestr(name, data)
            print('KVPress 0.5.4: metadata overrides: Transformers 5.6.0 and Fire >=0.7.1,<0.8', flush=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall',
                            '--no-deps', str(patched)], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install',
                            '-r', str(root / 'requirements.txt')], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', str(root / 'tests'), '-q'], check=True, cwd=root)


if __name__ == '__main__':
    main()
