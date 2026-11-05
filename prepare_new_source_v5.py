"""新整源对照 v5：中景/广角的真实后厨（针对"板在哪"补样本）。

抽样时刻先写死；随后人盲标，再看模型。选片原则：优先能看到操作台全貌的中景，而不是特写。
"""
import cv2
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'new_source_video_v5'
INBOX = BASE.parent / '待导入视频'

SOURCES = {
    'BV1pu411U7Zh': {'title': '餐厅后厨真实记录（4:40）',
                     'timestamps': [15, 40, 65, 90, 115, 140, 165, 190, 215, 240, 265]},
    'BV18pRhBgEyw': {'title': '千人园区食堂后厨（24:35）',
                     'timestamps': [60, 150, 240, 330, 420, 510, 600, 690, 780, 870, 960, 1050]},
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
        'plan': 'new_source_video_v5',
        'stage': 'frames_frozen_labels_pending',
        'sources': {k: v['title'] for k, v in SOURCES.items()},
        'locked_timestamps': {k: v['timestamps'] for k, v in SOURCES.items()},
        'selection_rule': '优先中景/广角操作台，避免特写（v3/v4 的教训：特写里板框≈整幅，位置信息少）',
        'model_inference_performed': False,
        'frames': records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('total', len(records))


if __name__ == '__main__':
    main()
