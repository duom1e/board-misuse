"""Train board_v4 (v2 + 15 verified Pexels hard negatives) and compare against v2.

Evaluation:
  1. Open Images board test (24 images, has ground truth) - did the original capability survive?
  2. Pexels hold-out (456 photos never used for labelling or training) - did the tabletop
     false positives drop? No ground truth there, so this is a behaviour count plus a
     contact sheet for visual confirmation.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA = BASE / 'board_dataset_v4'
PROBE = BASE / 'pexels_kitchen' / 'probe'
RUNS = BASE / 'training_runs'
RUN_NAME = 'board_v4_hardneg'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
CONF = 0.5


def counts(model, images):
    hits, boxes = 0, 0
    for path in images:
        result = model.predict(source=str(path), conf=CONF, imgsz=640, device=0, verbose=False)[0]
        number = 0 if result.boxes is None else len(result.boxes)
        boxes += number
        hits += 1 if number else 0
    return {'images': len(images), 'images_with_detection': hits, 'boxes': boxes}


def sheet(model, images, name, limit=12):
    font = ImageFont.truetype(str(FONT), 12)
    flagged = []
    for path in images:
        result = model.predict(source=str(path), conf=CONF, imgsz=640, device=0, verbose=False)[0]
        if result.boxes is not None and len(result.boxes):
            flagged.append((path, result))
    for index in range(0, min(len(flagged), limit), 12):
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, (path, result) in enumerate(flagged[index:index + 12]):
            image = Image.open(path).convert('RGB')
            draw = ImageDraw.Draw(image)
            for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                draw.rectangle(box, outline='blue', width=3)
                draw.text((box[0] + 2, box[1] + 2), f'{score:.2f}', fill='blue', font=font)
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
        canvas.save(PROBE / f'{name}_{index // 12 + 1}.jpg')
    return len(flagged)


def test_metrics(model):
    result = model.val(data=str(BASE / 'board_dataset' / 'data.yaml'), split='test', imgsz=640, batch=8,
                       device=0, workers=0, plots=False, verbose=False)
    return {k: float(v) for k, v in dict(result.results_dict).items()}


def main():
    holdout = [Path(line) for line in (DATA / 'pexels_holdout.txt').read_text().split() if line.strip()]
    print('holdout', len(holdout), flush=True)
    registry = json.loads((BASE / 'active_models.json').read_text())
    v2 = YOLO(registry['board']['weights'])
    before = counts(v2, holdout)
    before_test = test_metrics(v2)

    model = YOLO(registry['board']['weights'])
    model.train(data=str(DATA / 'data.yaml'), epochs=70, imgsz=640, batch=8, device=0, workers=0,
                amp=False, seed=42, deterministic=True, project=str(RUNS), name=RUN_NAME,
                exist_ok=True, patience=0, val=True, plots=False,
                lr0=0.0005, lrf=0.1, freeze=10, mosaic=0.2, scale=0.2, close_mosaic=10)
    tuned = YOLO(str(RUNS / RUN_NAME / 'weights' / 'best.pt'))
    after = counts(tuned, holdout)
    after_test = test_metrics(tuned)
    flagged = sheet(tuned, holdout, 'v4_flagged')
    sheet(v2, holdout, 'v2_flagged')
    payload = {
        'run': RUN_NAME, 'dataset': str(DATA),
        'holdout': {'images': len(holdout), 'note': 'the 24 verified probe images are in training; these are the rest'},
        'pexels_before': before, 'pexels_after': after,
        'board_test_before': before_test, 'board_test_after': after_test,
        'v4_flagged_images': flagged,
        'caveat': 'Pexels has no ground truth: counts are behaviour, precision needs the visual check on v4_flagged_*.jpg',
    }
    (BASE / 'board_v4_metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ('pexels_before', 'pexels_after', 'v4_flagged_images')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
