"""在"重置起点"的新基线上重跑样本增删实验。

用法（环境变量）：
  SWAP_BACKBONE  yolo11n.pt / yolo11s.pt
  SWAP_SEED      73 / 42 / 7 ...
  EXTRA_NEG      额外负例图片目录（空标签，可空）
  EXTRA_POS      额外正例目录（含 labels，可空）
  VARIANT_TAG    输出目录后缀，如 kitchenneg
基线始终为 pool4_positive15_seed73 的 270 条 train.txt。
"""
import json
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from train_multisource_paired15 import overlap

BASE = Path(__file__).resolve().parent
PAIRED = BASE / 'experiments' / 'multisource_paired15_20261111'
BASE_RUN = BASE / 'experiments' / 'pool4_positive15_seed73_20261115'
VIDEO = next((BASE.parent / '待导入视频').glob('*BV1bY4y1Y7v5*.mp4'))
CHICKEN_ANN = BASE / 'experiments' / 'chicken_block_full12_20261115' / 'annotations.json'
WHITE_ANN = BASE / 'experiments' / 'white_round_holdout_20261115' / 'annotations.json'

BACKBONE = os.environ.get('SWAP_BACKBONE', 'yolo11n.pt')
SEED = int(os.environ.get('SWAP_SEED', '73'))
EXTRA_NEG = os.environ.get('EXTRA_NEG', '')
EXTRA_POS = os.environ.get('EXTRA_POS', '')
TAG = os.environ.get('VARIANT_TAG', 'plain')
POS_REPEAT = int(os.environ.get('POS_REPEAT', '2'))
OUT = BASE / 'experiments' / ('rerun_%s_%s_seed%d_20261115'
                              % (BACKBONE.replace('.pt', ''), TAG, SEED))
CONF, IOU = 0.25, 0.5


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main():
    OUT.mkdir(exist_ok=False)
    train = (BASE_RUN / 'train.txt').read_text().splitlines()
    base_count = len([line for line in train if line.strip()])
    added = []
    if EXTRA_NEG:
        files = sorted((BASE / EXTRA_NEG).glob('*.jpg'))
        train += [str(p) for p in files]
        added += [str(p) for p in files]
    if EXTRA_POS:
        files = sorted((BASE / EXTRA_POS / 'images').glob('*.jpg'))
        train += [str(p) for p in files] * POS_REPEAT
        added += [str(p) for p in files]
    (OUT / 'train.txt').write_text('\n'.join(train) + '\n')
    (OUT / 'data.yaml').write_text('path: %s\ntrain: train.txt\nval: %s\nnames: [cutting_board]\n'
                                   % (OUT.as_posix(), (BASE / 'board_dataset/validation.txt').as_posix()))
    (OUT / 'plan.json').write_text(json.dumps({
        'backbone': BACKBONE, 'seed': SEED, 'base_entries': base_count,
        'entries': len([line for line in train if line.strip()]),
        'extra_neg': EXTRA_NEG, 'extra_pos': EXTRA_POS, 'added': added,
        'note': '基线与对照组完全相同，只多出额外样本',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    model = YOLO(BACKBONE)
    model.train(data=str(OUT / 'data.yaml'), epochs=15, batch=8, nbs=8, imgsz=640, workers=0, device=0,
                seed=SEED, optimizer='AdamW', lr0=.0001, warmup_epochs=0, amp=False,
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

    summary = {'backbone': BACKBONE, 'seed': SEED, 'tag': TAG, 'dev': dev,
               'chicken': anchors(CHICKEN_ANN), 'white': anchors(WHITE_ANN),
               'canteen': {'single': len(segments(1)), 'five': len(segments(5))}}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
