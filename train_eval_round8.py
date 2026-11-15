"""圆砧墩正例对照：8 张来自 2 个新来源的圆形菜墩，加入 270 条基线（不额外加负例）。

目的：开发集里 5 张"困难圆板"长期 0/5，看新来源的圆墩正例能否带来改善，
同时确认切鸡、白色圆板、食堂留出段不被拖累。
"""
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

from train_multisource_paired15 import overlap

BASE = Path(__file__).resolve().parent
PAIRED = BASE / 'experiments' / 'multisource_paired15_20261111'
SRC = BASE / 'experiments' / 'white_round_source_20261115' / 'frames'
POS = BASE / 'experiments' / 'round_block_positive_20261115'
OUT = BASE / 'experiments' / 'pool4_round8_20261115'
VIDEO = next((BASE.parent / '待导入视频').glob('*BV1bY4y1Y7v5*.mp4'))
CHICKEN_ANN = BASE / 'experiments' / 'chicken_block_full12_20261115' / 'annotations.json'
WHITE_ANN = BASE / 'experiments' / 'white_round_holdout_20261115' / 'annotations.json'
PICKS = [('BV12E411o7aX_4.jpg', [60, 120, 940, 1000]), ('BV12E411o7aX_5.jpg', [60, 120, 940, 1000]),
         ('BV12E411o7aX_6.jpg', [60, 120, 940, 1000]), ('BV12E411o7aX_7.jpg', [60, 120, 940, 1000]),
         ('BV1Ld4y1U7v9_1.jpg', [60, 150, 940, 1000]), ('BV1Ld4y1U7v9_2.jpg', [60, 150, 940, 1000]),
         ('BV1Ld4y1U7v9_5.jpg', [60, 150, 940, 1000]), ('BV1Ld4y1U7v9_6.jpg', [60, 150, 940, 1000])]
CONF, IOU = 0.25, 0.5


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def build():
    (POS / 'images').mkdir(parents=True, exist_ok=True)
    (POS / 'labels').mkdir(parents=True, exist_ok=True)
    files = []
    for name, box in PICKS:
        shutil.copyfile(SRC / name, POS / 'images' / name)
        (POS / 'labels' / (Path(name).stem + '.txt')).write_text(
            '0 %.6f %.6f %.6f %.6f\n' % ((box[0] + box[2]) / 2000, (box[1] + box[3]) / 2000,
                                         (box[2] - box[0]) / 1000, (box[3] - box[1]) / 1000), encoding='utf-8')
        files.append(str(POS / 'images' / name))
    return files


def main():
    positives = build()
    OUT.mkdir(exist_ok=False)
    base = (PAIRED / 'mixed.txt').read_text().splitlines()
    old_neg = sorted((BASE / 'experiments/wok_negative_review_20261112/images').glob('*.jpg'))
    old_neg += sorted((BASE / 'experiments/wood_counter_negative_20261112/images').glob('*.jpg'))
    extra = sorted((BASE / 'experiments/positive_pool_v4_20261112/images').glob('*.jpg'))
    train = base + [str(p) for p in old_neg] * 2 + [str(p) for p in extra] * 2 + positives
    (OUT / 'train.txt').write_text('\n'.join(train) + '\n')
    (OUT / 'data.yaml').write_text('path: %s\ntrain: train.txt\nval: %s\nnames: [cutting_board]\n'
                                   % (OUT.as_posix(), (BASE / 'board_dataset/validation.txt').as_posix()))
    (OUT / 'plan.json').write_text(json.dumps({'entries': len(train), 'positives': positives, 'repeat': 1,
                                               'baseline': 'pool4_positive15_seed73 (270)',
                                               'epochs': 15, 'imgsz': 640, 'seed': 73},
                                              ensure_ascii=False, indent=2), encoding='utf-8')

    model = YOLO(str(BASE / 'training_runs/board_v2_real/weights/best.pt'))
    model.train(data=str(OUT / 'data.yaml'), epochs=15, batch=8, nbs=8, imgsz=640, workers=0, device=0,
                seed=73, optimizer='AdamW', lr0=.0001, warmup_epochs=0, amp=False,
                mosaic=0, plots=False, project=str(OUT), name='run')
    tuned = YOLO(str(OUT / 'run' / 'weights' / 'last.pt'))

    cases = json.loads((PAIRED / 'plan.json').read_text())['cases']
    dev, hard, hard_total = [], 0, 0
    case_rows = []
    for threshold in (0.1, 0.25, 0.5):
        hits = false_frames = 0
        for case in cases:
            boxes = tuned.predict(case['image'], imgsz=640, rect=True, conf=.01, device=0, verbose=False)[0].boxes
            pairs = list(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()))
            hit = bool(case['gt']) and any(s >= threshold and overlap(b, case['gt']) >= IOU for b, s in pairs)
            if case['gt']:
                hits += hit
                case_rows.append({'frame': Path(case['image']).name, 'threshold': threshold, 'hit': hit})
            else:
                false_frames += any(s >= threshold for _, s in pairs)
        dev.append({'threshold': threshold, 'hits': hits, 'false_frames': false_frames})
    hard_frames = sorted({r['frame'] for r in case_rows})[:5]
    hard = sum(1 for r in case_rows if r['threshold'] == CONF and r['frame'] in hard_frames and r['hit'])
    hard_total = len(hard_frames)

    def anchors(path):
        rows = json.loads(path.read_text(encoding='utf-8'))
        out = []
        for threshold in (0.25, 0.5):
            hit = 0
            for row in rows:
                boxes = tuned.predict(row['image'], imgsz=640, rect=True, conf=.05, device=0, verbose=False)[0].boxes
                pairs = list(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()))
                hit += any(s >= threshold and iou(b, row['box']) >= IOU for b, s in pairs)
            out.append({'threshold': threshold, 'hits': hit, 'total': len(rows)})
        return out

    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(880 * fps))
    per_frame = []
    for _ in range(int(40 * fps)):
        ok, frame = capture.read()
        if not ok:
            break
        boxes = tuned.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0].boxes
        per_frame.append(boxes.xyxy.cpu().tolist() if boxes is not None and len(boxes) else [])
    capture.release()

    def segments(rule):
        segs, start = [], None
        for index, boxes in enumerate(per_frame):
            keep = False
            if boxes:
                if rule == 1:
                    keep = True
                elif index >= rule - 1:
                    keep = all(any(iou(boxes[0], other) >= IOU for other in win)
                               for win in per_frame[index - rule + 1:index])
            if keep and start is None:
                start = index
            if not keep and start is not None:
                segs.append([start, index - 1]); start = None
        if start is not None:
            segs.append([start, len(per_frame) - 1])
        return segs

    summary = {'dev': dev, 'first_five_positive_frames': hard_frames,
               'first_five_hits': '%d/%d' % (hard, hard_total),
               'chicken': anchors(CHICKEN_ANN), 'white': anchors(WHITE_ANN),
               'canteen': {'single': len(segments(1)), 'five': len(segments(5))},
               'baseline': {'dev_0.25': [9, 19], 'chicken_0.25': 11, 'white_0.25': 7, 'canteen': [5, 3]}}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
