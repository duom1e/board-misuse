"""新整源对照 v8：复用本机已有素材（woodkitchen / meat），补更多有板场景。

这两段此前从未进入任何 new_source_v* 批次，只在本机待导入目录里，等于零下载成本。
"""
import cv2
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'new_source_video_v8'
INBOX = BASE.parent / '待导入视频'

SOURCES = {
    'BV1sA411G7Yw': {'title': 'woodkitchen（57.4MB）',
                     'timestamps': [15, 45, 75, 105, 135, 165, 195, 225, 255, 285]},
    'BV14W41197Rd': {'title': 'meat（87.6MB）',
                     'timestamps': [20, 60, 100, 140, 180, 220, 260, 300, 340, 380]},
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
        'plan': 'new_source_video_v8',
        'stage': 'frames_frozen_labels_pending',
        'sources': {k: v['title'] for k, v in SOURCES.items()},
        'locked_timestamps': {k: v['timestamps'] for k, v in SOURCES.items()},
        'note': '复用本机已有素材，零下载成本；补更多有板场景',
        'model_inference_performed': False,
        'frames': records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('total', len(records))


if __name__ == '__main__':
    main()
