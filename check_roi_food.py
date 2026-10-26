"""Test the "known board region" design on real canteen frames.

If the board region is given (fixed camera or hand-drawn once), the only job left is food:
crop the region, upscale it, and run the food model there. This compares:
  A. whole frame at 640 (what the current prototype does)
  B. board region cropped and upscaled, food model only
on the same 40 real commercial-kitchen frames.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'real_canteen_probe'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
LEVELS = (0.5, 0.25)


def expand(box, width, height, margin=0.15):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy))


def main():
    registry = json.loads((BASE / 'active_models.json').read_text())
    board = YOLO(registry['board']['weights'])
    food = YOLO(registry['foodseg']['weights'])
    frames = sorted((PROBE / 'images').glob('*.jpg'))
    rows = []
    for path in frames:
        with Image.open(path) as handle:
            image = handle.convert('RGB')
        width, height = image.size
        proposal = board.predict(source=str(path), conf=0.20, imgsz=640, device=0, verbose=False)[0]
        boxes = proposal.boxes.xyxy.cpu().tolist() if proposal.boxes is not None and len(proposal.boxes) else []
        if not boxes:
            rows.append({'name': path.name, 'roi_source': 'none', 'roi_food': [], 'whole_food': []})
            continue
        # biggest proposal = the chopping block
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        roi = expand(box, width, height)
        crop = image.crop(tuple(int(v) for v in roi))
        scale = max(1.0, 640 / max(crop.width, 1))
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        roi_result = food.predict(source=crop, conf=min(LEVELS), imgsz=640, device=0, verbose=False)[0]
        whole_result = food.predict(source=str(path), conf=min(LEVELS), imgsz=640, device=0, verbose=False)[0]

        def items(result):
            if result.boxes is None or not len(result.boxes):
                return []
            return [{'category': food.names[int(cls)], 'conf': round(float(c), 3)}
                    for cls, c in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]

        rows.append({'name': path.name, 'roi_source': 'board_proposal', 'roi': [round(v, 1) for v in roi],
                     'roi_food': items(roi_result), 'whole_food': items(whole_result)})

    summary = {}
    for level in LEVELS:
        summary[f'roi_at_{level}'] = {
            'frames': len(rows),
            'frames_with_food': sum(1 for r in rows if any(d['conf'] >= level for d in r['roi_food'])),
            'meat_boxes': sum(1 for r in rows for d in r['roi_food'] if d['category'] == 'meat' and d['conf'] >= level),
        }
        summary[f'whole_at_{level}'] = {
            'frames_with_food': sum(1 for r in rows if any(d['conf'] >= level for d in r['whole_food'])),
            'meat_boxes': sum(1 for r in rows for d in r['whole_food'] if d['category'] == 'meat' and d['conf'] >= level),
        }
    summary['rows_with_roi'] = sum(1 for r in rows if r['roi_source'] == 'board_proposal')

    font = ImageFont.truetype(str(FONT), 12)
    canvas = Image.new('RGB', (960, 720), 'white')
    shown = [r for r in rows if r.get('roi')][:12]
    for slot, row in enumerate(shown):
        image = Image.open(PROBE / 'images' / row['name']).convert('RGB')
        draw = ImageDraw.Draw(image)
        draw.rectangle(row['roi'], outline='blue', width=4)
        image.thumbnail((320, 160))
        canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
        ImageDraw.Draw(canvas).text(((slot % 3) * 320 + 3, (slot // 3) * 180 + 3),
                                    row['name'][-10:-4], fill='yellow', font=font)
    canvas.save(PROBE / 'roi_examples.jpg')

    (PROBE / 'roi_food_check.json').write_text(json.dumps({
        'design': 'board region given (proposal or fixed camera), crop + upscale, food model inside',
        'probe': 'Chinese commercial kitchen cutting task, 40 frames',
        'summary': summary, 'rows': rows,
        'note': 'ROI comes from the board model here; in deployment it can be a fixed camera region.',
    }, indent=2, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
