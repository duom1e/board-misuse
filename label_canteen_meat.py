"""Colour-based meat proposals inside the chopping block, for labelling the canteen frames.

The food model detects nothing in this domain, so proposals come from colour instead: within
the board region, raw sliced pork is pink/red against warm wood. Proposals are only a starting
point - every frame is checked on a contact sheet before any box is used for training.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'real_canteen_probe'
LABELS = BASE / 'canteen_board_poscontrol' / 'labels.json'
OUT = BASE / 'canteen_food_labels'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
MIN_AREA_FRACTION = 0.004


def board_box(normalized, width, height):
    return (normalized[0] * width, normalized[1] * height, normalized[2] * width, normalized[3] * height)


def meat_boxes(image_bgr, roi):
    x1, y1, x2, y2 = (int(v) for v in roi)
    crop = image_bgr[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # raw pork: red/pink hues, moderate saturation; also catch paler cooked slices
    low = cv2.inRange(hsv, (0, 25, 45), (20, 255, 255))
    high = cv2.inRange(hsv, (160, 25, 45), (180, 255, 255))
    mask = cv2.morphologyEx(low | high, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    area_min = MIN_AREA_FRACTION * crop.shape[0] * crop.shape[1]
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < area_min or w < 12 or h < 12:
            continue
        boxes.append([x + x1, y + y1, x + w + x1, y + h + y1])
    return boxes


def main():
    OUT.mkdir(exist_ok=True)
    board = YOLO(json.loads((BASE / 'active_models.json').read_text())['board']['weights'])
    median = json.loads(LABELS.read_text())['median_box_normalized']
    frames = sorted((PROBE / 'images').glob('*.jpg'))
    records = []
    for path in frames:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        result = board.predict(source=str(path), conf=0.2, imgsz=960, device=0, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None and len(result.boxes) else []
        roi = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if boxes else board_box(median, width, height)
        proposals = meat_boxes(image, roi)
        records.append({'name': path.name, 'roi': [round(float(v), 1) for v in roi],
                        'proposals': [[round(float(v), 1) for v in b] for b in proposals]})
    font = ImageFont.truetype(str(FONT), 12)
    for start in range(0, len(records), 12):
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, record in enumerate(records[start:start + 12]):
            image = Image.open(PROBE / 'images' / record['name']).convert('RGB')
            draw = ImageDraw.Draw(image)
            draw.rectangle(record['roi'], outline='blue', width=3)
            for box in record['proposals']:
                draw.rectangle(box, outline='red', width=3)
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
            ImageDraw.Draw(canvas).text(((slot % 3) * 320 + 3, (slot // 3) * 180 + 3),
                                        f"{record['name'][-10:-4]} {len(record['proposals'])}",
                                        fill='yellow', font=font)
        canvas.save(OUT / f'proposal_sheet_{start // 12 + 1}.jpg')
    (OUT / 'proposals.json').write_text(json.dumps({
        'method': 'HSV red/pink threshold inside the board region, connected components >= 0.4% of the region',
        'status': 'proposals only - must be checked on proposal_sheet_*.jpg before training',
        'frames': records}, ensure_ascii=False, indent=1))
    counts = [len(r['proposals']) for r in records]
    print('frames', len(records), 'with proposal', sum(1 for c in counts if c), 'total boxes', sum(counts))


if __name__ == '__main__':
    main()
