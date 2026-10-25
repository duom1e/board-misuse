"""Near-duplicate check across dataset splits (dHash on a 9x8 grayscale thumbnail).

Same images appearing in train and test inflate test scores. If two different images are
the same scene re-shot, dHash <= 4 also flags them as suspected duplicates. This only
proves duplicates do not exist when it reports nothing - it cannot prove scenes are
independent, and it says nothing about annotation quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def dhash(path, size=8):
    with Image.open(path) as im:
        pixels = list(im.convert('L').resize((size + 1, size), Image.LANCZOS).tobytes())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset directory containing images/<split>/*.jpg')
    parser.add_argument('--threshold', type=int, default=4)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    splits = [d.name for d in (root / 'images').iterdir() if d.is_dir()]
    records = {}
    hashes = {}
    for split in splits:
        for path in sorted((root / 'images' / split).glob('*.jpg')):
            records[path] = {'split': split, 'dhash': dhash(path),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            hashes.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(path)
    exact = [{'sha256': key, 'files': [str(p) for p in paths], 'splits': sorted({records[p]['split'] for p in paths})}
             for key, paths in hashes.items() if len(paths) > 1]
    cross = []
    paths = list(records)
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            if records[left]['split'] == records[right]['split']:
                continue
            distance = bin(records[left]['dhash'] ^ records[right]['dhash']).count('1')
            if distance <= args.threshold:
                cross.append({'a': str(left), 'b': str(right), 'distance': distance,
                              'splits': [records[left]['split'], records[right]['split']]})
    payload = {'dataset': str(root), 'splits': {s: sum(1 for r in records.values() if r['split'] == s) for s in splits},
               'exact_duplicates': exact, 'cross_split_dhash_within_threshold': cross,
               'threshold': args.threshold,
               'limitation': 'Absence of flags does not prove scene independence or clean labels.'}
    out = Path(args.output) if args.output else root / 'similarity_check.json'
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({'splits': payload['splits'], 'exact_duplicates': len(exact),
                      'cross_split_suspects': len(cross), 'output': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
