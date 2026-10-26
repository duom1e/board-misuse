"""Positive control: can the board model learn this real canteen scene from 25 frames?

Train on frames 1-25, evaluate detection rate on frames 26-40 (same scene, later time), and also
re-check the Open Images board test to see whether the original capability survives.

This is NOT a generalisation result: one kitchen, one camera, one chef, same session.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA = BASE / 'canteen_board_poscontrol'
RUNS = BASE / 'training_runs'
RUN_NAME = 'board_canteen_poscontrol'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')


def detection_rate(model, images, conf, imgsz):
    hits, boxes = 0, 0
    for path in images:
        result = model.predict(source=str(path), conf=conf, imgsz=imgsz, device=0, verbose=False)[0]
        number = 0 if result.boxes is None else len(result.boxes)
        boxes += number
        hits += 1 if number else 0
    return {'images': len(images), 'images_with_detection': hits, 'boxes': boxes,
            'rate': round(hits / len(images), 3) if images else None}


def main():
    registry = json.loads((BASE / 'active_models.json').read_text())
    val_images = sorted((DATA / 'images' / 'val').glob('*.jpg'))
    reference = YOLO(registry['board']['weights'])
    before = {str(conf): detection_rate(reference, val_images, conf, 640) for conf in (0.5, 0.25)}
    before_test = {k: float(v) for k, v in dict(reference.val(
        data=str(BASE / 'board_dataset' / 'data.yaml'), split='test', imgsz=640, batch=8,
        device=0, workers=0, plots=False, verbose=False).results_dict).items()}

    model = YOLO(registry['board']['weights'])
    model.train(data=str(DATA / 'data.yaml'), epochs=120, imgsz=960, batch=4, device=0, workers=0,
                amp=False, seed=42, deterministic=True, project=str(RUNS), name=RUN_NAME,
                exist_ok=True, patience=0, val=True, plots=False,
                lr0=0.001, lrf=0.05, freeze=10, mosaic=0.0, scale=0.1, fliplr=0.0, close_mosaic=0)
    tuned = YOLO(str(RUNS / RUN_NAME / 'weights' / 'best.pt'))
    after = {str(conf): detection_rate(tuned, val_images, conf, 960) for conf in (0.5, 0.25)}
    after_test = {k: float(v) for k, v in dict(tuned.val(
        data=str(BASE / 'board_dataset' / 'data.yaml'), split='test', imgsz=640, batch=8,
        device=0, workers=0, plots=False, verbose=False).results_dict).items()}

    font = ImageFont.truetype(str(FONT), 12)
    for start in range(0, len(val_images), 12):
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, path in enumerate(val_images[start:start + 12]):
            image = Image.open(path).convert('RGB')
            result = tuned.predict(source=str(path), conf=0.25, imgsz=960, device=0, verbose=False)[0]
            draw = ImageDraw.Draw(image)
            if result.boxes is not None and len(result.boxes):
                for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    colour = 'blue' if score >= 0.5 else 'cyan'
                    draw.rectangle(box, outline=colour, width=3)
                    draw.text((box[0] + 2, box[1] + 2), f'{score:.2f}', fill=colour, font=font)
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
        canvas.save(DATA / f'val_pred_{start // 12 + 1}.jpg')

    payload = {
        'run': RUN_NAME, 'dataset': str(DATA),
        'scope': 'same-scene adaptation positive control: 25 train frames, 15 held-out frames from the same video',
        'frames_before': before, 'frames_after': after,
        'board_test_before': before_test, 'board_test_after': after_test,
        'note': 'all held-out frames contain the chopping block, so the rate is recall; extra boxes would be false positives',
    }
    (BASE / 'canteen_poscontrol_metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ('frames_before', 'frames_after')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
