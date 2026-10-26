"""导入视频：把「待导入视频」里的视频抽帧、跑模型、出对照表。

用法：
  python import_video.py                 # 处理待导入视频里的全部视频
  python import_video.py <视频路径>       # 处理指定文件
  python import_video.py --frames 60     # 指定抽帧数

输出到 outputs/imported_<视频名>/：
  frames/            抽出的帧
  detections.json    每帧的案板/食材检测结果
  sheet_*.jpg        带框对照表（蓝=案板≥0.5，青=0.25–0.5，红=食材）
  summary.json       统计
"""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
INBOX = BASE.parent / '待导入视频'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.flv', '.webm', '.m4v', '.ts'}


def extract(video, out_dir, count):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f'打不开视频：{video}')
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        raise RuntimeError(f'读不到帧数：{video}')
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i in range(count):
        index = int(total * (i + 0.5) / count)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        path = out_dir / f'frame_{index:06d}.jpg'
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved.append({'index': index, 'file': str(path)})
    capture.release()
    return {'frames_total': total, 'fps': fps, 'size': [width, height], 'saved': saved}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', nargs='?', default=None)
    parser.add_argument('--frames', type=int, default=40)
    args = parser.parse_args()

    if args.video:
        videos = [Path(args.video)]
    else:
        INBOX.mkdir(parents=True, exist_ok=True)
        videos = sorted(p for p in INBOX.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        print(f'待导入视频 里没有视频文件：{INBOX}')
        return

    registry = json.loads((BASE / 'active_models.json').read_text())
    board = YOLO(registry['board']['weights'])
    food = YOLO(registry['foodseg']['weights'])

    for video in videos:
        print(f'处理 {video.name} ...', flush=True)
        out_dir = BASE / 'imported' / video.stem
        info = extract(video, out_dir / 'frames', args.frames)
        records = []
        for frame in info['saved']:
            b = board.predict(source=frame['file'], conf=0.25, imgsz=960, device=0, verbose=False)[0]
            f = food.predict(source=frame['file'], conf=0.25, imgsz=960, device=0, verbose=False)[0]
            boards = [{'conf': round(float(c), 3), 'xyxy': [round(float(v), 1) for v in box]}
                      for box, c in zip(b.boxes.xyxy.cpu().tolist(), b.boxes.conf.cpu().tolist())] \
                if b.boxes is not None and len(b.boxes) else []
            foods = [{'category': food.names[int(cls)], 'conf': round(float(c), 3),
                      'xyxy': [round(float(v), 1) for v in box]}
                     for box, c, cls in zip(f.boxes.xyxy.cpu().tolist(), f.boxes.conf.cpu().tolist(),
                                            f.boxes.cls.cpu().tolist())] \
                if f.boxes is not None and len(f.boxes) else []
            records.append({'frame': frame['index'], 'file': frame['file'], 'boards': boards, 'foods': foods})

        font = ImageFont.truetype(str(FONT), 12)
        frames = [Path(r['file']) for r in records]
        for start in range(0, len(frames), 12):
            canvas = Image.new('RGB', (960, 720), 'white')
            for slot, record in enumerate(records[start:start + 12]):
                image = Image.open(record['file']).convert('RGB')
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
                                            f"f{record['frame']}", fill='yellow', font=font)
            canvas.save(out_dir / f'sheet_{start // 12 + 1}.jpg')

        summary = {
            'video': str(video), 'frames_total': info['frames_total'], 'fps': info['fps'],
            'size': info['size'], 'extracted': len(records),
            'frames_with_board_0.5': sum(1 for r in records if any(b['conf'] >= 0.5 for b in r['boards'])),
            'frames_with_board_0.25': sum(1 for r in records if r['boards']),
            'frames_with_food_0.5': sum(1 for r in records if any(x['conf'] >= 0.5 for x in r['foods'])),
            'frames_with_food_0.25': sum(1 for r in records if r['foods']),
            'board_model': registry['board']['weights'], 'food_model': registry['foodseg']['weights'],
            'note': '这是行为统计，不是准确率；请先看 sheet_*.jpg 再决定标哪几帧',
        }
        (out_dir / 'detections.json').write_text(json.dumps({'summary': summary, 'records': records},
                                                            ensure_ascii=False, indent=1))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f'输出目录：{out_dir}')


if __name__ == '__main__':
    main()
