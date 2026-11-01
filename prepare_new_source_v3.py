"""新整源对照 v3：砧板岗位 / 冷盘切配两段，冻结抽样与帧哈希（不含模型推理）。

抽样时刻先写死；随后由人盲标，再看模型。
"""
import cv2
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'new_source_video_v3'
INBOX = BASE.parent / '待导入视频'

SOURCES = {
    'BV12a41187Ja': {'title': '酒店砧板岗位实习（5:17）',
                     'timestamps': [15, 40, 65, 90, 115, 140, 165, 190, 215, 240, 265, 290]},
    'BV1pq4y147jk': {'title': '冷盘切配（7:20）',
                     'timestamps': [20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350, 380, 410]},
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
                            'size': [int(frame.shape[1]), int(frame.shape[0])], 'label': None})
        capture.release()
        print(bvid, 'frames', len([r for r in records if r['source_bvid'] == bvid]), flush=True)
    (OUT / 'frames_manifest.json').write_text(json.dumps({
        'plan': 'new_source_video_v3',
        'stage': 'frames_frozen_labels_pending',
        'sources': {k: v['title'] for k, v in SOURCES.items()},
        'locked_timestamps': {k: v['timestamps'] for k, v in SOURCES.items()},
        'model_inference_performed': False,
        'rule': 'labels must be recorded before any model sees these frames',
        'frames': records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('total', len(records))


if __name__ == '__main__':
    main()
