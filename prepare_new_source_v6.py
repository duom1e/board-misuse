"""新整源对照 v6：学校食堂（监管检查）× 食堂大锅炒土豆丝，冻结抽样与帧哈希。

选片目的：补**新场景**（此前 27 个板框过度集中在一两段视频），并优先中景。
"""
import cv2
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'new_source_video_v6'
INBOX = BASE.parent / '待导入视频'

SOURCES = {
    'BV1bY4y1Y7v5': {'title': '延安市场监管看小学食堂（24:48）',
                     'timestamps': [60, 180, 300, 420, 540, 660, 780, 900, 1020, 1140, 1260, 1380]},
    'BV163411f7cv': {'title': '食堂大锅炒土豆丝（1:08）',
                     'timestamps': [8, 16, 24, 32, 40, 48, 56]},
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
        'plan': 'new_source_video_v6',
        'stage': 'frames_frozen_labels_pending',
        'sources': {k: v['title'] for k, v in SOURCES.items()},
        'locked_timestamps': {k: v['timestamps'] for k, v in SOURCES.items()},
        'model_inference_performed': False,
        'frames': records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('total', len(records))


if __name__ == '__main__':
    main()
