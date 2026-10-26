"""Mix the 25 canteen frames into the normal board training set and see if both survive.

The pure canteen run lifted held-out canteen recall from 0.0 to 1.0 but destroyed the Open
Images board test (mAP50 0.912 -> 0.036). This run trains on board_v2 (242 images) plus the
25 canteen frames, then checks:
  1. held-out canteen frames (15, same scene) - recall should stay high
  2. Open Images board test (24) - should not collapse
"""
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
SRC = BASE / 'board_dataset_v2'
CANTEEN = BASE / 'canteen_board_poscontrol'
OUT = BASE / 'board_dataset_v5'
RUNS = BASE / 'training_runs'
RUN_NAME = 'board_v5_canteen_mix'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')


def link(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        dest.write_bytes(src.read_bytes())


def build():
    for split in ['train', 'validation', 'test']:
        for kind in ['images', 'labels']:
            for path in (SRC / kind / split).glob('*'):
                link(path, OUT / kind / split / path.name)
    for kind in ['images', 'labels']:
        for path in (CANTEEN / kind / 'train').glob('*'):
            link(path, OUT / kind / 'train' / path.name)
    lists = {}
    for split in ['train', 'validation', 'test']:
        paths = sorted((OUT / 'images' / split).glob('*.jpg'))
        (OUT / f'{split}.txt').write_text('\n'.join(p.as_posix() for p in paths) + '\n')
        lists[split] = len(paths)
    (OUT / 'data.yaml').write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n  0: cutting_board\n")
    print('dataset', lists, flush=True)
    return lists


def rate(model, images, conf, imgsz):
    hits = boxes = 0
    for path in images:
        result = model.predict(source=str(path), conf=conf, imgsz=imgsz, device=0, verbose=False)[0]
        number = 0 if result.boxes is None else len(result.boxes)
        hits += 1 if number else 0
        boxes += number
    return {'images': len(images), 'images_with_detection': hits, 'boxes': boxes,
            'rate': round(hits / len(images), 3) if images else None}


def main():
    lists = build()
    registry = json.loads((BASE / 'active_models.json').read_text())
    val_frames = sorted((CANTEEN / 'images' / 'val').glob('*.jpg'))
    model = YOLO(registry['board']['weights'])
    model.train(data=str(OUT / 'data.yaml'), epochs=70, imgsz=640, batch=8, device=0, workers=0,
                amp=False, seed=42, deterministic=True, project=str(RUNS), name=RUN_NAME,
                exist_ok=True, patience=0, val=True, plots=False,
                lr0=0.0005, lrf=0.1, freeze=10, mosaic=0.2, scale=0.2, close_mosaic=10)
    tuned = YOLO(str(RUNS / RUN_NAME / 'weights' / 'best.pt'))
    payload = {
        'run': RUN_NAME, 'dataset': str(OUT), 'images': lists,
        'canteen_frames_after': {str(c): rate(tuned, val_frames, c, 960) for c in (0.5, 0.25)},
        'board_test_after': {k: float(v) for k, v in dict(tuned.val(
            data=str(BASE / 'board_dataset' / 'data.yaml'), split='test', imgsz=640, batch=8,
            device=0, workers=0, plots=False, verbose=False).results_dict).items()},
        'note': 'canteen frames are the same scene as the pure-canteen run; open images test is the retention check',
    }
    font = ImageFont.truetype(str(FONT), 12)
    canvas = Image.new('RGB', (960, 720), 'white')
    for slot, path in enumerate(val_frames[:12]):
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
    canvas.save(OUT / 'canteen_mixed_predictions.jpg')
    (BASE / 'canteen_mix_metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
