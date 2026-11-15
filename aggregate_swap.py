"""汇总换底座实验：yolo11n(COCO 起步) vs yolo11s(COCO 起步)，各 5 个种子。"""
import json
import statistics as st
from pathlib import Path

BASE = Path(__file__).resolve().parent / 'experiments'
SEEDS = ('73', '42', '7', '11', '23')


def load(prefix, seed):
    candidates = [BASE / ('%s_seed%s_20261115' % (prefix, seed)),
                  BASE / ('%s_20261115' % prefix)]
    for folder in candidates:
        path = folder / 'summary.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
    return None


def row(data):
    dev = {r['threshold']: (r['hits'], r['false_frames']) for r in data['dev']}
    white = {r['threshold']: r['hits'] for r in data['white']}
    chicken = {r['threshold']: r['hits'] for r in data['chicken']}
    return {'hit25': dev[0.25][0], 'fp25': dev[0.25][1], 'hit50': dev[0.5][0], 'fp50': dev[0.5][1],
            'white': white[0.25], 'chicken': chicken[0.25],
            'canteen5': data['canteen']['five']}


def main():
    out = {}
    for prefix in ('swap_yolo11n', 'swap_yolo11s', 'base_swap_yolo11s'):
        rows = []
        for seed in SEEDS:
            data = load(prefix, seed)
            if data:
                rows.append((seed, row(data)))
        if rows:
            out[prefix] = rows

    for prefix, rows in out.items():
        print('==', prefix)
        for seed, r in rows:
            print('  seed %-3s hit25 %d fp25 %2d hit50 %d fp50 %d white %2d chicken %2d canteen5 %d'
                  % (seed, r['hit25'], r['fp25'], r['hit50'], r['fp50'], r['white'], r['chicken'], r['canteen5']))
        keys = ('hit25', 'fp25', 'hit50', 'fp50', 'white', 'chicken', 'canteen5')
        summary = {k: (round(st.mean([r[k] for _, r in rows]), 1),
                       min(r[k] for _, r in rows), max(r[k] for _, r in rows)) for k in keys}
        print('  mean(min-max):', {k: v for k, v in summary.items()})

    (BASE / 'swap_comparison.json').write_text(json.dumps(
        {p: [{'seed': s, **r} for s, r in rows] for p, rows in out.items()}, ensure_ascii=False, indent=2),
        encoding='utf-8')


if __name__ == '__main__':
    main()
