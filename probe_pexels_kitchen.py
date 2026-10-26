"""Run the current models on the Pexels kitchen photos (new, non-Open-Images source).

No ground-truth boxes exist here, so this measures behaviour: how often boards/food are
reported on real (mostly home) kitchen scenes, and boxes-per-image. Precision is estimated by
looking at contact sheets of the flagged images; recall is unknown.

Sources: pexels_kitchen/images (mixed), images2 (strict-slug survivors), canteen.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
ROOTS = {
    'mixed': BASE / 'pexels_kitchen' / 'images',
    'strict': BASE / 'pexels_kitchen' / 'images2',
    'canteen': BASE / 'pexels_kitchen' / 'canteen',
}
OUT = BASE / 'pexels_kitchen' / 'probe'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
CONF = 0.5


def main():
    OUT.mkdir(exist_ok=True)
    registry = json.loads((BASE / 'active_models.json').read_text())
    board = YOLO(registry['board']['weights'])
    food = YOLO(registry['foodseg']['weights'])
    records = []
    for group, root in ROOTS.items():
        images = sorted(root.glob('*.jpg'))
        for path in images:
            b = board.predict(source=str(path), conf=CONF, imgsz=640, device=0, verbose=False)[0]
            f = food.predict(source=str(path), conf=CONF, imgsz=640, device=0, verbose=False)[0]
            boards = [{'conf': round(float(c), 3), 'xyxy': [round(float(v), 1) for v in box]}
                      for box, c in zip(b.boxes.xyxy.cpu().tolist(), b.boxes.conf.cpu().tolist())] \
                if b.boxes is not None and len(b.boxes) else []
            foods = [{'category': food.names[int(cls)], 'conf': round(float(c), 3),
                      'xyxy': [round(float(v), 1) for v in box]}
                     for box, c, cls in zip(f.boxes.xyxy.cpu().tolist(), f.boxes.conf.cpu().tolist(),
                                            f.boxes.cls.cpu().tolist())] \
                if f.boxes is not None and len(f.boxes) else []
            records.append({'group': group, 'image': str(path), 'name': path.name,
                            'boards': boards, 'foods': foods})
        print(group, len(images), flush=True)

    summary = {}
    for group in ROOTS:
        rows = [r for r in records if r['group'] == group]
        summary[group] = {
            'images': len(rows),
            'images_with_board': sum(1 for r in rows if r['boards']),
            'images_with_food': sum(1 for r in rows if r['foods']),
            'board_boxes': sum(len(r['boards']) for r in rows),
            'food_boxes': sum(len(r['foods']) for r in rows),
        }
    total = {key: sum(summary[g][key] for g in summary) for key in ('images', 'images_with_board',
                                                                   'images_with_food', 'board_boxes', 'food_boxes')}

    flagged = [r for r in records if r['boards'] or r['foods']]
    font = ImageFont.truetype(str(FONT), 12)
    for index in range(0, min(len(flagged), 24), 12):
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, record in enumerate(flagged[index:index + 12]):
            image = Image.open(record['image']).convert('RGB')
            draw = ImageDraw.Draw(image)
            for box in record['boards']:
                draw.rectangle(box['xyxy'], outline='blue', width=3)
            for box in record['foods']:
                draw.rectangle(box['xyxy'], outline='red', width=3)
                draw.text((box['xyxy'][0] + 2, box['xyxy'][1] + 2), f"{box['category']} {box['conf']:.2f}",
                          fill='red', font=font)
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
            ImageDraw.Draw(canvas).text(((slot % 3) * 320 + 3, (slot // 3) * 180 + 3),
                                        record['name'][:14], fill='yellow', font=font)
        canvas.save(OUT / f'flagged_{index // 12 + 1}.jpg')

    (OUT / 'summary.json').write_text(json.dumps({
        'source': 'Pexels kitchen photos (free licence), new to every trained model',
        'confidence': CONF, 'imgsz': 640,
        'groups': summary, 'total': total,
        'images_flagged': len(flagged),
        'records': records,
        'note': 'No ground truth boxes here: these are behaviour counts. Precision needs the visual check on '
                'flagged_*.jpg; recall is unknown.',
    }, indent=2, ensure_ascii=False))
    print(json.dumps({'total': total, 'flagged': len(flagged)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
