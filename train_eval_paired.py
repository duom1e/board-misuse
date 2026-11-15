"""正负配对实验：270 条基线 + 15 张厨房负例 + 3 张木砧板正例（各 2 次）。

对照：pool4_positive15_seed73（基线）与 pool4_kitchenneg15（只加负例）。
只看三件事：开发集误报、切鸡/白板锚点是否比"只加负例"回来。
"""
import json
import shutil
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

from train_multisource_paired15 import overlap

BASE = Path(__file__).resolve().parent
PAIRED = BASE / 'experiments' / 'multisource_paired15_20261111'
NEG = BASE / 'experiments' / 'kitchen_neg2_20261115' / 'images'
SCAN = BASE / 'experiments' / 'scan_white_round_20261115' / 'frames'
POS = BASE / 'experiments' / 'wood_positive_20261115'
OUT = BASE / 'experiments' / 'pool4_paired_20261115'
VIDEO = next((BASE.parent / '待导入视频').glob('*BV1bY4y1Y7v5*.mp4'))
CHICKEN_ANN = BASE / 'experiments' / 'chicken_block_full12_20261115' / 'annotations.json'
WHITE_ANN = BASE / 'experiments' / 'white_round_holdout_20261115' / 'annotations.json'
POSITIVES = [('BV1yz4y1d7iF_20.jpg', [0, 0, 1000, 1000]),
             ('BV1Bt411Y7VF_20.jpg', [250, 100, 660, 920]),
             ('BV1Bt411Y7VF_80.jpg', [250, 110, 650, 900])]
CONF, IOU = 0.25, 0.5


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def build_positives():
    (POS / 'images').mkdir(parents=True, exist_ok=True)
    (POS / 'labels').mkdir(parents=True, exist_ok=True)
    files = []
    for name, box in POSITIVES:
        dst = POS / 'images' / name
        shutil.copyfile(SCAN / name, dst)
        with Image.open(dst) as im:
            im.size
        cx = (box[0] + box[2]) / 2 / 1000
        cy = (box[1] + box[3]) / 2 / 1000
        (POS / 'labels' / (dst.stem + '.txt')).write_text(
            '0 %.6f %.6f %.6f %.6f\n' % (cx, cy, (box[2] - box[0]) / 1000, (box[3] - box[1]) / 1000),
            encoding='utf-8')
        files.append(str(dst))
    return files


def main():
    positives = build_positives()
    negatives = [str(p) for p in sorted(NEG.glob('*.jpg'))]
    OUT.mkdir(exist_ok=False)
    base = (PAIRED / 'mixed.txt').read_text().splitlines()
    old_neg = sorted((BASE / 'experiments/wok_negative_review_20261112/images').glob('*.jpg'))
    old_neg += sorted((BASE / 'experiments/wood_counter_negative_20261112/images').glob('*.jpg'))
    extra = sorted((BASE / 'experiments/positive_pool_v4_20261112/images').glob('*.jpg'))
    train = base + [str(p) for p in old_neg] * 2 + [str(p) for p in extra] * 2 + negatives + positives * 2
    (OUT / 'train.txt').write_text('\n'.join(train) + '\n')
    (OUT / 'data.yaml').write_text(
        'path: %s\ntrain: train.txt\nval: %s\nnames: [cutting_board]\n'
        % (OUT.as_posix(), (BASE / 'board_dataset/validation.txt').as_posix()))
    (OUT / 'plan.json').write_text(json.dumps({
        'entries': len(train), 'kitchen_negatives': len(negatives), 'wood_positives': positives,
        'compare': ['pool4_positive15_seed73_20261115 (270)', 'pool4_kitchenneg15_20261115 (285)'],
        'epochs': 15, 'imgsz': 640, 'seed': 73,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    model = YOLO(str(BASE / 'training_runs/board_v2_real/weights/best.pt'))
    model.train(data=str(OUT / 'data.yaml'), epochs=15, batch=8, nbs=8, imgsz=640, workers=0, device=0,
                seed=73, optimizer='AdamW', lr0=.0001, warmup_epochs=0, amp=False,
                mosaic=0, plots=False, project=str(OUT), name='run')
    tuned = YOLO(str(OUT / 'run' / 'weights' / 'last.pt'))

    cases = json.loads((PAIRED / 'plan.json').read_text())['cases']
    dev = []
    for threshold in (0.1, 0.25, 0.5):
        hits = false_frames = 0
        for case in cases:
            boxes = tuned.predict(case['image'], imgsz=640, rect=True, conf=.01, device=0, verbose=False)[0].boxes
            pairs = list(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()))
            if case['gt']:
                hits += any(s >= threshold and overlap(b, case['gt']) >= IOU for b, s in pairs)
            else:
                false_frames += any(s >= threshold for _, s in pairs)
        dev.append({'threshold': threshold, 'hits': hits, 'false_frames': false_frames})

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

    summary = {'dev': dev, 'chicken_anchors': anchors(CHICKEN_ANN), 'white_anchors': anchors(WHITE_ANN),
               'canteen_holdout': {'single': len(segments(1)), 'five': len(segments(5)),
                                   'frames_with_any': sum(1 for b in per_frame if b)},
               'baseline': {'dev_0.25': [9, 19], 'chicken_0.25': 11, 'white_0.25': 7},
               'neg_only': {'dev_0.25': [6, 7], 'chicken_0.25': 7, 'white_0.25': 2}}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
