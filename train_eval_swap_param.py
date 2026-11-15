"""换底座对照：同样的 270 条数据、同样 15 轮、同种子，把 yolo11n 换成 yolo11s（本机已有权重，无需下载）。

目的：验证"数据规模见顶"还是"底座容量见顶"。
四组评估与基线完全一致：开发集 14 正 / 70 负、切鸡 12 锚点、白色圆板 12 锚点、
食堂巡检 880–920s 的候选段（单帧 / 连续5帧）。
"""
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from train_multisource_paired15 import overlap

BASE = Path(__file__).resolve().parent
PAIRED = BASE / 'experiments' / 'multisource_paired15_20261111'
BASE_RUN = BASE / 'experiments' / 'pool4_positive15_seed73_20261115'
SEED = int(__import__('os').environ.get('SWAP_SEED','73'))
BACKBONE = __import__('os').environ.get('SWAP_BACKBONE','yolo11s.pt')
OUT = BASE / 'experiments' / ('swap_%s_seed%d_20261115' % (BACKBONE.replace('.pt',''), SEED))
VIDEO = next((BASE.parent / '待导入视频').glob('*BV1bY4y1Y7v5*.mp4'))
CHICKEN_ANN = BASE / 'experiments' / 'chicken_block_full12_20261115' / 'annotations.json'
WHITE_ANN = BASE / 'experiments' / 'white_round_holdout_20261115' / 'annotations.json'
CONF, IOU = 0.25, 0.5


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main():
    OUT.mkdir(exist_ok=False)
    train_list = (BASE_RUN / 'train.txt').read_text()
    (OUT / 'train.txt').write_text(train_list)
    (OUT / 'data.yaml').write_text('path: %s\ntrain: train.txt\nval: %s\nnames: [cutting_board]\n'
                                   % (OUT.as_posix(), (BASE / 'board_dataset/validation.txt').as_posix()))
    entries = len([line for line in train_list.splitlines() if line.strip()])
    (OUT / 'plan.json').write_text(json.dumps({
        'backbone': BACKBONE, 'entries': entries, 'epochs': 15, 'imgsz': 640, 'seed': SEED,
        'data': '与 pool4_positive15_seed73 完全相同（270 条）',
        'baseline': {'dev_0.25': [9, 19], 'chicken_0.25': 11, 'white_0.25': 7, 'canteen': [5, 3]},
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

    summary = {'backbone': BACKBONE, 'dev': dev, 'chicken': anchors(CHICKEN_ANN),
               'white': anchors(WHITE_ANN),
               'canteen': {'single': len(segments(1)), 'five': len(segments(5))},
               'baseline_yolo11n': {'dev_0.25': [9, 19], 'chicken_0.25': 11, 'white_0.25': 7,
                                    'canteen': [5, 3]}}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
