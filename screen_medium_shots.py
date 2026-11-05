"""从已下载的学校食堂视频里**密集采样**，专门找"中景 + 有板"的画面。

此前 v6 只在该视频取了 12 个点（60–1380s），全是检查流程。这次每 60 秒取一帧，
覆盖整个 24 分钟，目的不是再测一次，而是**筛出可用于标注的中景有板帧**。
"""
import cv2
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'screen_medium_v1'
INBOX = BASE.parent / '待导入视频'
BVID = 'BV1bY4y1Y7v5'


def main():
    video = next(INBOX.glob(f'*{BVID}*.mp4'), None)
    if video is None:
        raise SystemExit('未找到视频')
    (OUT / 'frames').mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps else 0
    records = []
    for seconds in range(15, int(duration), 60):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
        ok, frame = capture.read()
        if not ok:
            continue
        name = f'{BVID}_{seconds:04d}s.jpg'
        path = OUT / 'frames' / name
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        records.append({'seconds': seconds, 'file': str(path),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    capture.release()
    (OUT / 'manifest.json').write_text(json.dumps({'video': str(video), 'fps': fps,
                                                   'duration_seconds': round(duration, 1),
                                                   'frames': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('frames', len(records), 'duration', round(duration))


if __name__ == '__main__':
    main()
