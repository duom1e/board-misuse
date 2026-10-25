"""Fine-tune the board detector on verified real-kitchen boards and hard negatives.

Starts from the existing board baseline so the Open Images board knowledge is kept, then adds
2 verified real boards and 44 verified no-board scenes. Evaluates three things:
  1. board_dataset test split (does the original capability survive?)
  2. detection counts on the 92 held-out kitchen images (old model vs new model)
  3. contact sheets of both models on that hold-out for visual verification
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA = BASE / 'board_dataset_v2'
PROBE = BASE / 'real_kitchen_probe'
RUNS = BASE / 'training_runs'
RUN_NAME = 'board_v2_real'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
CONF = 0.5


def holdout_images():
    labels = json.loads((BASE / 'board_real_labels.json').read_text())
    used = set(labels['no_board']) | set(labels['board_accept_box']) | {
        row['image'] for row in labels['skip_ambiguous']}
    images = []
    for path in sorted((PROBE / 'images').glob('*.jpg')):
        if any(path.stem.startswith(prefix) for prefix in used):
            continue
        images.append(path)
    return images


def counts(model, images, conf=CONF):
    hits = 0
    boxes = 0
    for path in images:
        result = model.predict(source=str(path), conf=conf, imgsz=640, device=0, verbose=False)[0]
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
    return len(flagged)


def main():
    images = holdout_images()
    print('holdout images', len(images), flush=True)
    baseline = YOLO(str(RUNS / 'board_baseline' / 'weights' / 'best.pt'))
    before = counts(baseline, images)
    before_test = None
    val_before = baseline.val(data=str((BASE / 'board_dataset' / 'data.yaml')), split='test', imgsz=640, batch=8,
                              device=0, workers=0, plots=False, verbose=False)
    before_test = {k: float(v) for k, v in dict(val_before.results_dict).items()}

    model = YOLO(str(RUNS / 'board_baseline' / 'weights' / 'best.pt'))
    model.train(data=str(DATA / 'data.yaml'), epochs=60, imgsz=640, batch=8, device=0, workers=0,
                amp=False, seed=42, deterministic=True, project=str(RUNS), name=RUN_NAME,
                exist_ok=True, patience=0, val=True, plots=False,
                lr0=0.0005, lrf=0.1, freeze=10, mosaic=0.2, scale=0.2, close_mosaic=10)
    tuned = YOLO(str(RUNS / RUN_NAME / 'weights' / 'best.pt'))
    after = counts(tuned, images)
    val_after = tuned.val(data=str((BASE / 'board_dataset' / 'data.yaml')), split='test', imgsz=640, batch=8,
                          device=0, workers=0, plots=False, verbose=False)
    after_test = {k: float(v) for k, v in dict(val_after.results_dict).items()}
    sheet(baseline, images, 'board_holdout_before.jpg')
    flagged_after = sheet(tuned, images, 'board_holdout_after.jpg')
    payload = {
        'run': RUN_NAME, 'dataset': str(DATA), 'holdout_images': len(images),
        'holdout': 'the 92 kitchen images not used for labelling or training',
        'board_dataset_test_before': before_test, 'board_dataset_test_after': after_test,
        'kitchen_holdout_before': before, 'kitchen_holdout_after': after,
        'sheets': ['board_holdout_before.jpg', 'board_holdout_after.jpg'],
        'flagged_images_after': flagged_after,
        'note': 'Counts are detections at conf 0.5, not accuracy. Precision still needs the visual check on the sheets.',
    }
    (BASE / 'board_v2_metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
