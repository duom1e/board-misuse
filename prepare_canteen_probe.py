"""Extract evenly spaced frames from the Chinese commercial kitchen cutting video.

Source: https://huggingface.co/datasets/nova-dynamics/Chinese_Commercial_Kitchen_Manipulation_Dataset_Preview
Licence: CC BY-NC 4.0 (local research only; do not publish images or derived weights).

Every frame of this task shows the chef cutting on the thick round chopping block, so it can
serve as a real-canteen recall probe for the board model (board present in all frames).
"""
import json
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
VIDEO = BASE.parent / 'work' / 'datasets' / 'cn_commercial_kitchen' / 'overhead_cutting.mp4'
OUT = BASE / 'real_canteen_probe'
FRAMES = 40


def main():
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f'cannot open {VIDEO}')
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'frames={total} fps={fps:.2f} size={width}x{height} seconds={total / max(fps, 1):.1f}', flush=True)
    indices = [int(total * (i + 0.5) / FRAMES) for i in range(FRAMES)]
    saved = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        path = OUT / 'images' / f'cnkitchen_{index:06d}.jpg'
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved.append({'frame_index': index, 'file': str(path)})
    capture.release()
    (OUT / 'manifest.json').write_text(json.dumps({
        'source': 'nova-dynamics/Chinese_Commercial_Kitchen_Manipulation_Dataset_Preview, task_01_cutting/overhead',
        'licence': 'CC BY-NC 4.0 - local non-commercial research only, do not publish',
        'video': {'frames': total, 'fps': fps, 'size': [width, height]},
        'extracted': len(saved), 'frames': saved,
        'scene': 'professional chef cutting meat on a thick round wooden chopping block, stainless kitchen',
        'note': 'the chopping block is present in every frame, so board detection rate here is a recall probe',
    }, indent=2, ensure_ascii=False))
    print('saved', len(saved))


if __name__ == '__main__':
    main()
