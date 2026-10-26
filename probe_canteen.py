"""Run the board and food models on the real commercial-kitchen frames.

Every frame shows the chef cutting on the thick round chopping block, so:
  - board detection rate = recall proxy for "finds the board"
  - boxes drawn on a contact sheet let a human check whether the box is on the block
Food: the chef is cutting raw meat, so meat detections should be frequent and correct.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'real_canteen_probe'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
LEVELS = (0.5, 0.25)


def main():
    registry = json.loads((BASE / 'active_models.json').read_text())
    board = YOLO(registry['board']['weights'])
    food = YOLO(registry['foodseg']['weights'])
    frames = sorted((PROBE / 'images').glob('*.jpg'))
    records = []
    for path in frames:
        b = board.predict(source=str(path), conf=min(LEVELS), imgsz=640, device=0, verbose=False)[0]
        f = food.predict(source=str(path), conf=min(LEVELS), imgsz=640, device=0, verbose=False)[0]
        boards = [{'conf': round(float(c), 3), 'xyxy': [round(float(v), 1) for v in box]}
                  for box, c in zip(b.boxes.xyxy.cpu().tolist(), b.boxes.conf.cpu().tolist())] \
            if b.boxes is not None and len(b.boxes) else []
        foods = [{'category': food.names[int(cls)], 'conf': round(float(c), 3),
                  'xyxy': [round(float(v), 1) for v in box]}
                 for box, c, cls in zip(f.boxes.xyxy.cpu().tolist(), f.boxes.conf.cpu().tolist(),
                                        f.boxes.cls.cpu().tolist())] \
            if f.boxes is not None and len(f.boxes) else []
        records.append({'image': str(path), 'name': path.name, 'boards': boards, 'foods': foods})

    summary = {}
    for level in LEVELS:
        summary[str(level)] = {
            'frames': len(records),
            'frames_with_board': sum(1 for r in records if any(d['conf'] >= level for d in r['boards'])),
            'frames_with_food': sum(1 for r in records if any(d['conf'] >= level for d in r['foods'])),
            'meat_boxes': sum(1 for r in records for d in r['foods'] if d['category'] == 'meat' and d['conf'] >= level),
            'vegetable_boxes': sum(1 for r in records for d in r['foods'] if d['category'] == 'vegetable' and d['conf'] >= level),
        }

    font = ImageFont.truetype(str(FONT), 12)
    for start in range(0, len(records), 12):
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, record in enumerate(records[start:start + 12]):
            image = Image.open(record['image']).convert('RGB')
            draw = ImageDraw.Draw(image)
            for box in record['boards']:
                colour = 'blue' if box['conf'] >= 0.5 else 'cyan'
                draw.rectangle(box['xyxy'], outline=colour, width=3)
                draw.text((box['xyxy'][0] + 2, box['xyxy'][1] + 2), f"b {box['conf']:.2f}", fill=colour, font=font)
            for box in record['foods']:
                draw.rectangle(box['xyxy'], outline='red', width=2)
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
            ImageDraw.Draw(canvas).text(((slot % 3) * 320 + 3, (slot // 3) * 180 + 3),
                                        record['name'][-10:-4], fill='yellow', font=font)
        canvas.save(PROBE / f'sheet_{start // 12 + 1}.jpg')

    (PROBE / 'results.json').write_text(json.dumps({
        'probe': 'Chinese commercial kitchen, cutting task, overhead camera',
        'licence': 'CC BY-NC 4.0, local research only',
        'board_present_in_all_frames': True,
        'summary': summary, 'records': records,
        'note': 'board detection rate is a recall proxy, not precision; food correctness needs the visual check.',
    }, indent=2, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
