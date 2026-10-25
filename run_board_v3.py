"""Train board_dataset_v3 and compare baseline / v2 / v3 on the fresh kitchen probe."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA = BASE / 'board_dataset_v3'
PROBE = BASE / 'real_kitchen_probe2'
RUNS = BASE / 'training_runs'
RUN_NAME = 'board_v3_real'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
CONF = 0.5


def detect(model, images):
    hits, boxes = 0, 0
    for path in images:
        result = model.predict(source=str(path), conf=CONF, imgsz=640, device=0, verbose=False)[0]
        number = 0 if result.boxes is None else len(result.boxes)
        boxes += number
        hits += 1 if number else 0
    return {'images': len(images), 'images_with_detection': hits, 'boxes': boxes}


def sheet(model, images, name, limit=12):
    font = ImageFont.truetype(str(FONT), 13)
    flagged = []
    for path in images:
        result = model.predict(source=str(path), conf=CONF, imgsz=640, device=0, verbose=False)[0]
        if result.boxes is not None and len(result.boxes):
            flagged.append((path, result))
    flagged = flagged[:limit]
    canvas = Image.new('RGB', (960, 720), 'white')
    for slot, (path, result) in enumerate(flagged):
        image = Image.open(path).convert('RGB')
        draw = ImageDraw.Draw(image)
        for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            draw.rectangle(box, outline='blue', width=3)
            draw.text((box[0] + 2, box[1] + 2), f'{score:.2f}', fill='blue', font=font)
        image.thumbnail((320, 160))
        canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
    canvas.save(PROBE / name)
    return flagged


def test_metrics(model):
    result = model.val(data=str(BASE / 'board_dataset' / 'data.yaml'), split='test', imgsz=640, batch=8,
                       device=0, workers=0, plots=False, verbose=False)
    return {k: float(v) for k, v in dict(result.results_dict).items()}


def main():
    images = sorted((PROBE / 'images').glob('*.jpg'))
    baseline = YOLO(str(RUNS / 'board_baseline' / 'weights' / 'best.pt'))
    v2 = YOLO(str(RUNS / 'board_v2_real' / 'weights' / 'best.pt'))
    before = {'baseline': detect(baseline, images), 'v2': detect(v2, images)}

    model = YOLO(str(RUNS / 'board_baseline' / 'weights' / 'best.pt'))
    model.train(data=str(DATA / 'data.yaml'), epochs=70, imgsz=640, batch=8, device=0, workers=0,
                amp=False, seed=42, deterministic=True, project=str(RUNS), name=RUN_NAME,
                exist_ok=True, patience=0, val=True, plots=False,
                lr0=0.0005, lrf=0.1, freeze=10, mosaic=0.2, scale=0.2, close_mosaic=10)
    tuned = YOLO(str(RUNS / RUN_NAME / 'weights' / 'best.pt'))
    after = detect(tuned, images)
    sheet(v2, images, 'fresh_probe_v2.jpg')
    flagged = sheet(tuned, images, 'fresh_probe_v3.jpg')
    payload = {
        'run': RUN_NAME,
        'fresh_probe': {'path': str(PROBE), 'images': len(images),
                        'note': 'downloaded after all earlier probes were consumed by training'},
        'detections_before': before, 'detections_after': after,
        'board_dataset_test': {'before': test_metrics(baseline), 'after': test_metrics(tuned)},
        'flagged_images_after': len(flagged),
        'sheets': ['fresh_probe_v2.jpg', 'fresh_probe_v3.jpg'],
        'caveat': 'counts are behaviour at conf 0.5; precision needs the visual check on the sheets',
    }
    (BASE / 'board_v3_metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
