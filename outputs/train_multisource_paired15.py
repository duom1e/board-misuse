from pathlib import Path
import hashlib
import json
import os
import itertools
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments/multisource_paired15_20261111'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

def main():
    os.environ['YOLO_OFFLINE'] = 'True'
    OUT.mkdir(exist_ok=False)
    original = (BASE / 'board_dataset/train.txt').read_text().splitlines()
    manifest = json.loads((BASE / 'experiments/board_multisource_bbox_v1/manifest.json').read_text())
    supplement = [r['target'] for r in manifest['records'] if r['split'] == 'train' and '475s' not in r['image']]
    for folder in ['board_chicken_round_v1', 'dark_chopping_round_v1', 'board_counter_round_v1', 'roast_mid_board_v1']:
        supplement.extend(json.loads((BASE / 'experiments' / folder / 'supplement.json').read_text()))
    supplement = list(dict.fromkeys(supplement))
    cases = json.loads((BASE / 'experiments/roast_mid_audit_v1/real.json').read_text())
    held = json.loads((BASE / 'experiments/roast_source_check_20261111/plan.json').read_text())['cases']
    cases.extend({'image': r['image'], 'gt': r['box']} for r in held)
    evaluation_hashes = {digest(r['image']) for r in cases}
    assert not evaluation_hashes.intersection(digest(p) for p in original + supplement), 'Exact image train/evaluation overlap'
    assert all('BV1aR4y1Z7FR' not in p and 'BV1pu411U7Zh' not in p and 'BV13s411H7dn' not in p for p in supplement)
    extra_count = 2 * len(supplement)
    plan = {'epochs': 15, 'supplement': supplement, 'supplement_unique': len(supplement), 'entries_per_arm': len(original)+extra_count, 'excluded': 'BV1bqsfzbELh_475s: woven mat mislabel', 'evaluation_count': len(cases), 'limitations': 'Previously viewed development data; approximate single reviewer labels; base source identities incomplete; single seed.', 'cases': cases}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    summary = []
    for arm, extras in [('real', list(itertools.islice(itertools.cycle(original), extra_count))), ('mixed', supplement * 2)]:
        (OUT / f'{arm}.txt').write_text('\n'.join(original + extras)+'\n', encoding='utf-8')
        config = OUT / f'{arm}.yaml'
        config.write_text(f'path: {OUT.as_posix()}\ntrain: {arm}.txt\nval: {(BASE / "board_dataset/validation.txt").as_posix()}\nnames: [cutting_board]\n', encoding='utf-8')
        model = YOLO(str(BASE / 'training_runs/board_v2_real/weights/best.pt'))
        model.train(data=str(config), epochs=15, batch=8, nbs=8, imgsz=640, workers=0, device=0, seed=42, optimizer='AdamW', lr0=.0001, warmup_epochs=0, amp=False, mosaic=0, plots=False, project=str(OUT), name=arm+'_run')
        weight = OUT / f'{arm}_run/weights/last.pt'
        detector = YOLO(str(weight))
        rows = []
        for case in cases:
            boxes = detector.predict(case['image'], imgsz=960, rect=True, conf=.01, device=0, verbose=False)[0].boxes
            predictions = [{'box': b, 'score': c, 'iou': overlap(b, case['gt']) if case['gt'] else None} for b,c in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist())]
            rows.append({'image': case['image'], 'gt': case['gt'], 'boxes': predictions})
        (OUT / f'{arm}_predictions.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        for threshold in [.1, .25, .5]:
            hits = sum(any(b['score'] >= threshold and b['iou'] >= .5 for b in r['boxes']) for r in rows if r['gt'])
            false = sum(any(b['score'] >= threshold for b in r['boxes']) for r in rows if not r['gt'])
            summary.append({'arm': arm, 'threshold': threshold, 'positive_hits': hits, 'positive_total': sum(bool(r['gt']) for r in rows), 'negative_false': false, 'negative_total': sum(r['gt'] is None for r in rows), 'weight_sha256': digest(weight)})
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))

if __name__ == '__main__':
    main()
