"""复现候选模型（board_neg15_v1）的四组指标：开发集 / 切鸡 / 白板 / 食堂段。

用法：python verify_board_neg15.py [imgsz]
输出：training_runs/board_neg15_v1/verification_<imgsz>.json
"""
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from train_multisource_paired15 import overlap

BASE = Path(__file__).resolve().parent
RUN = BASE / 'training_runs' / 'board_neg15_v1'
PAIRED = BASE / 'experiments' / 'multisource_paired15_20261111'
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
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 640
    model = YOLO(str(RUN / 'weights' / 'last.pt'))
    cases = json.loads((PAIRED / 'plan.json').read_text())['cases']

    dev = []
    for threshold in (0.25, 0.5):
        hits = false_frames = 0
        for case in cases:
            boxes = model.predict(case['image'], imgsz=size, rect=True, conf=.01, device=0, verbose=False)[0].boxes
            pairs = list(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()))
            if case['gt']:
                hits += any(s >= threshold and overlap(b, case['gt']) >= IOU for b, s in pairs)
            else:
                false_frames += any(s >= threshold for _, s in pairs)
        dev.append({'threshold': threshold, 'hits': hits, 'positive_total': 14,
                    'false_frames': false_frames, 'negative_total': 70})

    def anchors(path):
        rows = json.loads(path.read_text(encoding='utf-8'))
        out = []
        for threshold in (0.25, 0.5):
            hit = 0
            for row in rows:
                boxes = model.predict(row['image'], imgsz=size, rect=True, conf=.05, device=0, verbose=False)[0].boxes
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
        boxes = model.predict(frame, imgsz=size, conf=CONF, device=0, verbose=False)[0].boxes
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

    result = {'imgsz': size, 'dev': dev, 'chicken': anchors(CHICKEN_ANN), 'white': anchors(WHITE_ANN),
              'canteen': {'single': len(segments(1)), 'five': len(segments(5))}}
    (RUN / f'verification_{size}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
