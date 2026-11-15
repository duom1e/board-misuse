"""用**第二来源**（BV18pRhBgEyw）的锅具/托盘误报帧做负例，同条件对照。

为什么重做：上一轮把排水负例取自评估片段本身（BV1bY4y1Y7v5 880–920s），属于在评估片段上训练，
结论无效。本轮的负例全部来自另一段食堂视频，BV1bY4y1Y7v5 该片段保持为**未参与训练的留出**。

负例选择：16 张候选逐张核对后，剔除 495s（疑似绿色砧板，不确定），其余 15 张确认为锅/盆/托盘/人员，
无砧板 → 复制为 images/ + 空 labels/。
"""
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from train_multisource_paired15 import overlap

BASE = Path(__file__).resolve().parent
PAIRED = BASE / 'experiments' / 'multisource_paired15_20261111'
SRC = BASE / 'experiments' / 'drain_negative_src2_20261115' / 'images'
NEG = BASE / 'experiments' / 'kitchen_neg2_20261115'
OUT = BASE / 'experiments' / 'pool4_kitchenneg15_20261115'
VIDEO = next((BASE.parent / '待导入视频').glob('*BV1bY4y1Y7v5*.mp4'))
CHICKEN_ANN = BASE / 'experiments' / 'chicken_block_full12_20261115' / 'annotations.json'
WHITE_ANN = BASE / 'experiments' / 'white_round_holdout_20261115' / 'annotations.json'
EXCLUDE = {'cand_0495s.jpg'}
CONF, IOU = 0.25, 0.5


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def build_negatives():
    (NEG / 'images').mkdir(parents=True, exist_ok=True)
    (NEG / 'labels').mkdir(parents=True, exist_ok=True)
    kept = []
    for path in sorted(SRC.glob('*.jpg')):
        if path.name in EXCLUDE:
            continue
        target = NEG / 'images' / path.name
        shutil.copyfile(path, target)
        (NEG / 'labels' / (path.stem + '.txt')).write_text('', encoding='utf-8')
        kept.append(str(target))
    return kept


def main():
    import cv2

    kept = build_negatives()
    OUT.mkdir(exist_ok=False)
    base = (PAIRED / 'mixed.txt').read_text().splitlines()
    negatives = sorted((BASE / 'experiments/wok_negative_review_20261112/images').glob('*.jpg'))
    negatives += sorted((BASE / 'experiments/wood_counter_negative_20261112/images').glob('*.jpg'))
    extra = sorted((BASE / 'experiments/positive_pool_v4_20261112/images').glob('*.jpg'))
    train = base + [str(p) for p in negatives] * 2 + [str(p) for p in extra] * 2 + kept
    (OUT / 'train.txt').write_text('\n'.join(train) + '\n')
    (OUT / 'data.yaml').write_text(
        f'path: {OUT.as_posix()}\ntrain: train.txt\nval: {(BASE / "board_dataset/validation.txt").as_posix()}\nnames: [cutting_board]\n')
    (OUT / 'plan.json').write_text(json.dumps({
        'entries': len(train), 'kitchen_negatives': kept, 'repeat': 1,
        'baseline': 'experiments/pool4_positive15_seed73_20261115（270 条、34 批）',
        'holdout': 'BV1bY4y1Y7v5 880–920s 未参与训练',
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
        dev.append({'threshold': threshold, 'hits': hits, 'false_frames': false_frames,
                    'positive_total': 14, 'negative_total': 70})

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

    summary = {
        'dev': dev, 'chicken_anchors': anchors(CHICKEN_ANN), 'white_anchors': anchors(WHITE_ANN),
        'canteen_holdout': {'frames': len(per_frame), 'single_frame_segments': len(segments(1)),
                            'five_frame_segments': len(segments(5)),
                            'frames_with_any': sum(1 for b in per_frame if b)},
        'baseline': {'dev_0.25': '9 hits / 19 FP', 'chicken_0.25': '11/12', 'white_0.25': '7/12',
                     'canteen': '4–5 single-frame segments / 3 five-frame segments'},
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
