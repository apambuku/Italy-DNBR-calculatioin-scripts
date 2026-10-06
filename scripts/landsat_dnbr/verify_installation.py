"""Check publication integrity, installed versions and exact ancillary inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, help='Directory laid out as documented in README.md')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append('The tested Python version is 3.12; this interpreter is ' + sys.version.split()[0])
    with (root / 'MANIFEST.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            path = root / row['path']
            if not path.is_file() or sha256(path) != row['sha256']:
                errors.append('Missing or modified publication file: ' + row['path'])
    for line in (root / 'requirements.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, expected = line.strip().split('==')
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append('Missing dependency: ' + name)
            continue
        if actual != expected:
            errors.append(f'Dependency differs from tested version: {name} {actual}; expected {expected}')
    if args.input_root:
        for row in json.loads((root / 'input_manifest.json').read_text()):
            path = args.input_root / row['path']
            if not path.is_file() or sha256(path) != row['sha256']:
                errors.append('Missing or different reference input: ' + row['path'])
    for error in errors:
        print('FAIL:', error)
    if errors:
        raise SystemExit(1)
    print('PASS: publication checksums and direct dependency versions match.')
    if args.input_root:
        print('PASS: all ancillary inputs match the reference checksums.')
    else:
        print('Inputs not checked; use --input-root as documented in README.md.')
    print('This is an installation/input check, not an end-to-end processing test or an Earth Engine authentication check.')


if __name__ == '__main__':
    main()
