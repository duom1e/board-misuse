"""Positive control: fine-tune the board model on the real canteen frames, same scene.

Because the camera looks down at one cutting station, the chopping block sits in a narrow
band of the frame. This first measures how stable the model's own proposals are, then writes
one consistent box per frame (median of the proposals, widened) so the labels are not noisy.

Split by time: first 25 frames train, last 15 frames evaluation. That is a same-scene
adaptation test, NOT a generalisation result.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'real_canteen_probe'
OUT = BASE / 'canteen_board_poscontrol'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
TRAIN_FRAMES = 25


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    registry = json.loads((BASE / 'active_models.json').read_text())
    model = YOLO(registry['board']['weights'])
    frames = sorted((PROBE / 'images').glob('*.jpg'))
    proposals = []
    for path in frames:
        with Image.open(path) as handle:
            width, height = handle.size
        result = model.predict(source=str(path), conf=0.1, imgsz=960, device=0, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None and len(result.boxes) else []
        boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]
        proposals.append({'name': path.name, 'boxes': [[round(v, 1) for v in b] for b in boxes],
                          'size': [width, height]})

    usable = [p for p in proposals if p['boxes']]
    print(f'frames={len(proposals)} with proposal={len(usable)}', flush=True)
    if not usable:
        raise SystemExit('no proposals at all')
    norm = np.array([[b[0] / p['size'][0], b[1] / p['size'][1], b[2] / p['size'][0], b[3] / p['size'][1]]
                     for p in usable for b in p['boxes'][:1]])
    median = np.median(norm, axis=0)
    # per-frame label = the proposal that best matches the median region; outliers fall back to median
    chosen = {}
    for record in proposals:
        width, height = record['size']
        best, best_iou = None, -1.0
        for candidate in record['boxes']:
            norm_box = [candidate[0] / width, candidate[1] / height, candidate[2] / width, candidate[3] / height]
            value = iou(norm_box, list(median))
            if value > best_iou:
                best, best_iou = norm_box, value
        if best is None or best_iou < 0.3:
            chosen[record['name']] = list(median)
        else:
            chosen[record['name']] = best
    ious = [max(iou([b[0] / p['size'][0], b[1] / p['size'][1], b[2] / p['size'][0], b[3] / p['size'][1]],
                    list(median)) for b in p['boxes']) if p['boxes'] else 0.0 for p in proposals]
    print('IoU of best proposal vs median: min %.2f median %.2f' % (min(ious), float(np.median(ious))), flush=True)

    pad_x, pad_y = 0.02, 0.03
    box = [max(0.0, median[0] - pad_x), max(0.0, median[1] - pad_y),
           min(1.0, median[2] + pad_x), min(1.0, median[3] + pad_y)]
    OUT.mkdir(exist_ok=True)
    (OUT / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (OUT / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (OUT / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (OUT / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(frames):
        split = 'train' if index < TRAIN_FRAMES else 'val'
        with Image.open(path) as handle:
            image = handle.convert('RGB')
        image.save(OUT / 'images' / split / path.name, quality=92)
        width, height = image.size
        frame_box = chosen.get(path.name, list(median))
        cx, cy = (frame_box[0] + frame_box[2]) / 2, (frame_box[1] + frame_box[3]) / 2
        bw, bh = frame_box[2] - frame_box[0], frame_box[3] - frame_box[1]
        (OUT / 'labels' / split / f'{path.stem}.txt').write_text(
            f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
    for split in ['train', 'val']:
        paths = sorted((OUT / 'images' / split).glob('*.jpg'))
        (OUT / f'{split}.txt').write_text('\n'.join(p.as_posix() for p in paths) + '\n')
    (OUT / 'data.yaml').write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: cutting_board\n")

    font = ImageFont.truetype(str(FONT), 13)
    canvas = Image.new('RGB', (960, 540), 'white')
    for slot, path in enumerate(frames[:6]):
        image = Image.open(OUT / 'images' / ('train' if slot < 6 else 'val') / path.name).convert('RGB')
        draw = ImageDraw.Draw(image)
        width, height = image.size
        draw.rectangle((box[0] * width, box[1] * height, box[2] * width, box[3] * height), outline='lime', width=4)
        image.thumbnail((320, 160))
        canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
    canvas.save(OUT / 'label_preview.jpg')
    (OUT / 'labels.json').write_text(json.dumps({
        'method': 'one consistent box per frame = median of the board model proposals (conf 0.1) widened, '
                  'because the camera looks at a single cutting station',
        'median_box_normalized': [round(float(v), 4) for v in box],
        'proposal_iou_vs_median': {'min': round(min(ious), 3), 'median': round(float(np.median(ious)), 3)},
        'frames_train': TRAIN_FRAMES, 'frames_val': len(frames) - TRAIN_FRAMES,
        'caveat': 'same scene adaptation, not generalisation; labels are one fixed region, not per-frame hand drawing',
    }, indent=2, ensure_ascii=False))
    print('READY', TRAIN_FRAMES, len(frames) - TRAIN_FRAMES)


if __name__ == '__main__':
    main()
