"""后处理过滤：候选框落在画面下缘（地面/排水区）就丢弃，看能否压掉排水沟误报。

不动模型、不训练。基线模型 = pool4_positive15_seed73（270 条）。
三组评估：
  ① 食堂巡检 880–920s：单帧候选段 / 连续5帧候选段（过滤前 vs 过滤后）
  ② 切鸡 12 锚点、③ 白色圆板 12 锚点：命中率过滤前后是否变化
过滤规则（先固定）：框中心 y 坐标 > 画面高度 * RATIO 视为地面候选。
"""
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'floor_filter_20261115'
WEIGHTS = BASE / 'experiments' / 'pool4_positive15_seed73_20261115' / 'run' / 'weights' / 'last.pt'
VIDEO = next((BASE.parent / '待导入视频').glob('*BV1bY4y1Y7v5*.mp4'))
CHICKEN_ANN = BASE / 'experiments' / 'chicken_block_full12_20261115' / 'annotations.json'
WHITE_ANN = BASE / 'experiments' / 'white_round_holdout_20261115' / 'annotations.json'
CONF, IOU = 0.25, 0.5
RATIOS = (1.01, 0.85, 0.8, 0.75)


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def center_ok(box, height, ratio):
    return ((box[1] + box[3]) / 2) <= height * ratio


def main():
    OUT.mkdir(exist_ok=True)
    model = YOLO(str(WEIGHTS))

    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(880 * fps))
    per_frame = []
    for _ in range(int(40 * fps)):
        ok, frame = capture.read()
        if not ok:
            break
        boxes = model.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0].boxes
        per_frame.append(boxes.xyxy.cpu().tolist() if boxes is not None and len(boxes) else [])
    capture.release()

    def segments(frames, rule):
        segs, start = [], None
        for index, boxes in enumerate(frames):
            keep = False
            if boxes:
                if rule == 1:
                    keep = True
                elif index >= rule - 1:
                    keep = all(any(iou(boxes[0], other) >= IOU for other in win)
                               for win in frames[index - rule + 1:index])
            if keep and start is None:
                start = index
            if not keep and start is not None:
                segs.append([start, index - 1]); start = None
        if start is not None:
            segs.append([start, len(frames) - 1])
        return segs

    canteen = []
    for ratio in RATIOS:
        filtered = [[b for b in boxes if center_ok(b, height, ratio)] for boxes in per_frame]
        canteen.append({'ratio': ratio,
                        'single': len(segments(filtered, 1)), 'five': len(segments(filtered, 5)),
                        'frames_with_any': sum(1 for boxes in filtered if boxes)})

    def anchors(path):
        rows = json.loads(path.read_text(encoding='utf-8'))
        out = []
        for ratio in RATIOS:
            hit = 0
            for row in rows:
                image = cv2.imread(row['image'])
                h = image.shape[0]
                boxes = model.predict(image, imgsz=640, rect=True, conf=.05, device=0, verbose=False)[0].boxes
                pairs = list(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()))
                hit += any(s >= CONF and iou(b, row['box']) >= IOU and center_ok(b, h, ratio) for b, s in pairs)
            out.append({'ratio': ratio, 'hits': hit, 'total': len(rows)})
        return out

    summary = {'canteen': canteen, 'chicken': anchors(CHICKEN_ANN), 'white': anchors(WHITE_ANN),
               'note': 'ratio=1.01 等价于不过滤；ratio 越小过滤越狠'}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
