"""Run the board and food models on the real-kitchen probe and summarise what happens."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'real_kitchen_probe'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')


def main():
    from ultralytics import YOLO
    registry = json.loads((BASE / 'active_models.json').read_text())
    board = YOLO(str(BASE / 'training_runs' / 'board_baseline' / 'weights' / 'best.pt'))
    food = YOLO(registry['foodseg']['weights'])
    records = []
    for path in sorted((PROBE / 'images').glob('*.jpg')):
        board_result = board.predict(source=str(path), conf=0.25, imgsz=640, device=0, verbose=False)[0]
        food_result = food.predict(source=str(path), conf=0.25, imgsz=640, device=0, verbose=False)[0]

        def items(result, model):
            if result.boxes is None or not len(result.boxes):
                return []
            return [{'category': model.names[int(cls)], 'confidence': round(float(conf), 4),
                     'xyxy': [round(float(v), 1) for v in box]}
                    for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                              result.boxes.conf.cpu().tolist(),
                                              result.boxes.cls.cpu().tolist())]

        record = {'image': path.name, 'boards': items(board_result, board), 'foods': items(food_result, food)}
        record['boards_at_0.5'] = [b for b in record['boards'] if b['confidence'] >= 0.5]
        record['foods_at_0.5'] = [d for d in record['foods'] if d['confidence'] >= 0.5]
        records.append(record)

    summary = {
        'images': len(records),
        'images_with_board_at_0.5': sum(1 for r in records if r['boards_at_0.5']),
        'images_with_food_at_0.5': sum(1 for r in records if r['foods_at_0.5']),
        'images_with_food_at_0.25': sum(1 for r in records if r['foods']),
        'images_with_any_food_or_board_at_0.5': sum(1 for r in records if r['boards_at_0.5'] or r['foods_at_0.5']),
        'food_detections_by_category_at_0.5': {},
    }
    for record in records:
        for detection in record['foods_at_0.5']:
            key = detection['category']
            summary['food_detections_by_category_at_0.5'][key] = summary['food_detections_by_category_at_0.5'].get(key, 0) + 1
    summary['note'] = ('Real kitchen scenes from Open Images validation/test, never trained on. Counts are behaviour, '
                       'not accuracy: correctness was checked by eye only for the images placed on the contact sheets.')
    (PROBE / 'domain_probe.json').write_text(json.dumps({'summary': summary, 'images': records}, indent=2, ensure_ascii=False))

    flagged = [r for r in records if r['boards_at_0.5'] or r['foods_at_0.5']]
    font = ImageFont.truetype(str(FONT), 14)
    for sheet_index in range(0, min(len(flagged), 24), 12):
        sheet = Image.new('RGB', (960, 720), 'white')
        for slot, record in enumerate(flagged[sheet_index:sheet_index + 12]):
            image = Image.open(PROBE / 'images' / record['image']).convert('RGB')
            draw = ImageDraw.Draw(image)
            for board_box in record['boards_at_0.5']:
                draw.rectangle(board_box['xyxy'], outline='blue', width=3)
            for food_box in record['foods_at_0.5']:
                draw.rectangle(food_box['xyxy'], outline='red', width=3)
                draw.text((food_box['xyxy'][0] + 2, food_box['xyxy'][1] + 2),
                          f"{food_box['category']} {food_box['confidence']:.2f}", fill='red', font=font)
            image.thumbnail((320, 160))
            sheet.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
        sheet.save(PROBE / f'contact_sheet_{sheet_index // 12 + 1}.jpg')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('sheets', len(range(0, min(len(flagged), 24), 12)), 'flagged', len(flagged))


if __name__ == '__main__':
    main()
