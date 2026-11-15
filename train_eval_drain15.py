"""排水格栅负例对照：同条件训练 + 与基线一致的四组评估。

设计（跑之前固定，写进 plan.json）：
  · 训练：与 pool4_positive15_seed73 完全相同的数据与超参（270 条、15 轮、AdamW 1e-4、seed 73、640），
    只额外加入 **4 张排水沟/地砖/炒锅线** 的已核对无板帧各重复 2 次（+8 条 → 278 条）；
  · 评估：①开发集 14 正 / 70 负（640，conf .1/.25/.5）
          ②切鸡 12 个锚点、③白色圆板 12 个锚点（640，conf .25/.5，IoU≥.5 计命中）
          ④食堂巡检 880–920s 全段：conf .25 下的候选段数（单帧规则 / 连续 5 帧规则）
判据：负例误报下降、同时切鸡与白板命中不明显下降；否则不推广。
"""
import json
from pathlib import Path

from ultralytics import YOLO

from train_multisource_paired15 import overlap

BASE = Path(__file__).resolve().parent
PAIRED = BASE / 'experiments' / 'multisource_paired15_20261111'
OUT = BASE / 'experiments' / 'pool4_drain15_20261115'
DRAIN = BASE / 'experiments' / 'drain_negative_20261115' / 'images'
VIDEO = next((BASE.parent / '待导入视频').glob('*BV1bY4y1Y7v5*.mp4'))
CHICKEN_ANN = BASE / 'experiments' / 'chicken_block_full12_20261115' / 'annotations.json'
WHITE_ANN = BASE / 'experiments' / 'white_round_holdout_20261115' / 'annotations.json'
CONF = 0.25
IOU = 0.5


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main():
    import cv2

    OUT.mkdir(exist_ok=False)
    base = (PAIRED / 'mixed.txt').read_text().splitlines()
    negatives = sorted((BASE / 'experiments/wok_negative_review_20261112/images').glob('*.jpg'))
    negatives += sorted((BASE / 'experiments/wood_counter_negative_20261112/images').glob('*.jpg'))
    assert len(negatives) == 10
    extra = sorted((BASE / 'experiments/positive_pool_v4_20261112/images').glob('*.jpg'))
    assert len(extra) == 7
    drain = sorted(DRAIN.glob('*.jpg'))
    train = base + [str(p) for p in negatives] * 2 + [str(p) for p in extra] * 2 + [str(p) for p in drain] * 2
    (OUT / 'train.txt').write_text('\n'.join(train) + '\n')
    config = OUT / 'data.yaml'
    config.write_text(f'path: {OUT.as_posix()}\ntrain: train.txt\nval: {(BASE / "board_dataset/validation.txt").as_posix()}\nnames: [cutting_board]\n')
    (OUT / 'plan.json').write_text(json.dumps({
        'entries': len(train), 'base_entries': len(base) + len(negatives) * 2 + len(extra) * 2,
        'drain_negatives': [str(p) for p in drain], 'drain_repeat': 2,
        'epochs': 15, 'imgsz': 640, 'seed': 73, 'optimizer': 'AdamW', 'lr0': 0.0001,
        'baseline': 'experiments/pool4_positive15_seed73_20261115（270 条、34 批）',
        'note': '本轮 278 条，批次数比基线多；命中/误报差异不能单独归因',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    model = YOLO(str(BASE / 'training_runs/board_v2_real/weights/best.pt'))
    model.train(data=str(config), epochs=15, batch=8, nbs=8, imgsz=640, workers=0, device=0,
                seed=73, optimizer='AdamW', lr0=.0001, warmup_epochs=0, amp=False,
                mosaic=0, plots=False, project=str(OUT), name='run')
    tuned = YOLO(str(OUT / 'run' / 'weights' / 'last.pt'))

    # ① 开发集
    cases = json.loads((PAIRED / 'plan.json').read_text())['cases']
    dev = []
    for threshold in (0.1, 0.25, 0.5):
        hits = false_frames = 0
        for case in cases:
            boxes = tuned.predict(case['image'], imgsz=640, rect=True, conf=.01, device=0, verbose=False)[0].boxes
            pairs = list(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()))
            if case['gt']:
                if any(score >= threshold and overlap(box, case['gt']) >= IOU for box, score in pairs):
                    hits += 1
            else:
                if any(score >= threshold for _, score in pairs):
                    false_frames += 1
        dev.append({'threshold': threshold, 'hits': hits, 'positive_total': 14,
                    'false_frames': false_frames, 'negative_total': 70})

    # ②③ 锚点
    def anchors(path, key='box'):
        rows = json.loads(path.read_text(encoding='utf-8'))
        result = []
        for threshold in (0.25, 0.5):
            hit = 0
            for row in rows:
                boxes = tuned.predict(row['image'], imgsz=640, rect=True, conf=.05, device=0, verbose=False)[0].boxes
                pairs = list(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()))
                if any(score >= threshold and iou(box, row['box']) >= IOU for box, score in pairs):
                    hit += 1
            result.append({'threshold': threshold, 'hits': hit, 'total': len(rows)})
        return result

    chicken = anchors(CHICKEN_ANN)
    white = anchors(WHITE_ANN)

    # ④ 食堂巡检 880–920s：单帧段数 vs 连续5帧
    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(880 * fps))
    frames = int(40 * fps)
    per_frame = []
    for _ in range(frames):
        ok, frame = capture.read()
        if not ok:
            break
        boxes = tuned.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0].boxes
        per_frame.append(boxes.xyxy.cpu().tolist() if boxes is not None and len(boxes) else [])
    capture.release()

    def segments(rule):
        segs = []
        start_index = None
        for index, boxes in enumerate(per_frame):
            keep = False
            if boxes:
                best = boxes[0]
                if rule == 1:
                    keep = True
                elif index >= rule - 1:
                    window = per_frame[index - rule + 1:index]
                    keep = all(any(iou(best, other) >= IOU for other in win) for win in window)
            if keep and start_index is None:
                start_index = index
            if not keep and start_index is not None:
                segs.append([start_index, index - 1]); start_index = None
        if start_index is not None:
            segs.append([start_index, len(per_frame) - 1])
        return segs

    canteen = {'frames': len(per_frame),
               'single_frame_segments': len(segments(1)),
               'five_frame_segments': len(segments(5)),
               'frames_with_any': sum(1 for boxes in per_frame if boxes)}

    summary = {'dev': dev, 'chicken_anchors': chicken, 'white_anchors': white, 'canteen': canteen,
               'baseline_dev': 'hits 9/6 (0.25/0.5), false 19/8',
               'baseline_chicken_0.25': '11/12 anchors (single frame)',
               'baseline_white_0.25': '7/12 anchors',
               'baseline_canteen': '4-5 segments single frame, 3 with five-frame rule'}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
