"""真实后厨备餐段探针：从 B 站视频里取指定区间，抽帧、跑模型、出对照表。

用法：python probe_canteen2.py <视频> [--start 300] [--end 900] [--frames 40]
"""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
LEVELS = (0.5, 0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0, help='0 表示到视频结尾')
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--name', default=None)
    args = parser.parse_args()

    video = Path(args.video)
    name = args.name or video.stem[:40].replace(' ', '_')
    out = BASE / 'imported' / f'{name}_prep'
    (out / 'frames').mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    end = args.end or total
    start = max(0, args.start)
    span = max(1, end - start)
    registry = json.loads((BASE / 'active_models.json').read_text())
    board = YOLO(registry['board']['weights'])
    food = YOLO(registry['foodseg']['weights'])

    records = []
    for i in range(args.frames):
        index = start + int(span * (i + 0.5) / args.frames)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        path = out / 'frames' / f'frame_{index:06d}.jpg'
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        b = board.predict(source=str(path), conf=min(LEVELS), imgsz=960, device=0, verbose=False)[0]
        f = food.predict(source=str(path), conf=min(LEVELS), imgsz=960, device=0, verbose=False)[0]
        boards = [{'conf': round(float(c), 3), 'xyxy': [round(float(v), 1) for v in box]}
                  for box, c in zip(b.boxes.xyxy.cpu().tolist(), b.boxes.conf.cpu().tolist())] \
            if b.boxes is not None and len(b.boxes) else []
        foods = [{'category': food.names[int(cls)], 'conf': round(float(c), 3),
                  'xyxy': [round(float(v), 1) for v in box]}
                 for box, c, cls in zip(f.boxes.xyxy.cpu().tolist(), f.boxes.conf.cpu().tolist(),
                                        f.boxes.cls.cpu().tolist())] \
            if f.boxes is not None and len(f.boxes) else []
        records.append({'frame': index, 'file': str(path), 'boards': boards, 'foods': foods})
    capture.release()

    font = ImageFont.truetype(str(FONT), 12)
    for start_index in range(0, len(records), 12):
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, record in enumerate(records[start_index:start_index + 12]):
            image = Image.open(record['file']).convert('RGB')
            draw = ImageDraw.Draw(image)
            for box in record['boards']:
                colour = 'blue' if box['conf'] >= 0.5 else 'cyan'
                draw.rectangle(box['xyxy'], outline=colour, width=3)
                draw.text((box['xyxy'][0] + 2, box['xyxy'][1] + 2), f"b {box['conf']:.2f}", fill=colour, font=font)
            for box in record['foods']:
                draw.rectangle(box['xyxy'], outline='red', width=2)
                draw.text((box['xyxy'][0] + 2, box['xyxy'][1] + 14), f"{box['category']}", fill='red', font=font)
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
            ImageDraw.Draw(canvas).text(((slot % 3) * 320 + 3, (slot // 3) * 180 + 3),
                                        f"f{record['frame']}", fill='yellow', font=font)
        canvas.save(out / f'sheet_{start_index // 12 + 1}.jpg')

    summary = {
        'video': str(video), 'segment': [start, end], 'frames_total': total,
        'extracted': len(records),
        'board_at_0.5': sum(1 for r in records if any(b['conf'] >= 0.5 for b in r['boards'])),
        'board_at_0.25': sum(1 for r in records if r['boards']),
        'food_at_0.5': sum(1 for r in records if any(x['conf'] >= 0.5 for x in r['foods'])),
        'food_at_0.25': sum(1 for r in records if r['foods']),
        'note': '行为统计，不是准确率；先看 sheet_*.jpg',
    }
    (out / 'detections.json').write_text(json.dumps({'summary': summary, 'records': records},
                                                    ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('输出目录', out)


if __name__ == '__main__':
    main()
