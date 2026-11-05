"""新整源对照 v7：操作台前但**没有砧板**的边界负例。

目标不是再补有板样本，而是补"最难判的无板画面"——人在操作台前干活、台面是裸不锈钢，
没有砧板。此前 v6 的负例多远离操作台，难度不对等。
"""
import cv2
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'new_source_video_v7'
INBOX = BASE.parent / '待导入视频'

SOURCES = {
    'BV1b1ixB5ERA': {'title': '井然有序的后厨现场（1:57，炒锅线）',
                     'timestamps': [8, 20, 32, 44, 56, 68, 80, 92, 104, 116]},
    'BV1PW411F7mD': {'title': 'kitchennew（42.8MB，待看内容）',
                     'timestamps': [20, 45, 70, 95, 120, 145, 170, 195, 220, 245]},
}


def find_video(bvid):
    for path in INBOX.glob(f'*{bvid}*.mp4'):
        return path
    return None


def main():
    (OUT / 'frames').mkdir(parents=True, exist_ok=True)
    records = []
    for bvid, spec in SOURCES.items():
        video = find_video(bvid)
        if video is None:
            print('missing', bvid)
            continue
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for seconds in spec['timestamps']:
            index = int(round(seconds * fps))
            if index >= total:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            name = f'{bvid}_{seconds:03d}s.jpg'
            path = OUT / 'frames' / name
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            records.append({'source_bvid': bvid, 'seconds': seconds, 'frame_index': index,
                            'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'size': [int(frame.shape[1]), int(frame.shape[0])]})
        capture.release()
        print(bvid, len([r for r in records if r['source_bvid'] == bvid]), flush=True)
    (OUT / 'frames_manifest.json').write_text(json.dumps({
        'plan': 'new_source_video_v7',
        'stage': 'frames_frozen_labels_pending',
        'sources': {k: v['title'] for k, v in SOURCES.items()},
        'locked_timestamps': {k: v['timestamps'] for k, v in SOURCES.items()},
        'model_inference_performed': False,
        'frames': records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('total', len(records))


if __name__ == '__main__':
    main()
