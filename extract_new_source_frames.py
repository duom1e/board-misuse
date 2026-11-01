"""按 new_source_video_plan_v1.json 抽帧并冻结（不含任何模型推理）。

目的：在模型看到这批帧之前，把来源、时刻、SHA 全部固定下来，随后由人先标注。
GPU 此刻被训练占用，本脚本只用 opencv 读帧，CPU 即可。
"""
import hashlib
import json
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
PLAN = BASE / 'new_source_video_plan_v1.json'
OUT = BASE / 'experiments' / 'new_source_video_v1'


def main():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    frames_dir = OUT / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in plan['sources']:
        video = Path(source['file'])
        if not video.exists():
            print('missing', video)
            continue
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for seconds in plan['sampling']['timestamps_seconds']:
            index = int(round(seconds * fps))
            if index >= total:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            name = f"{source['bvid']}_{seconds:03d}s.jpg"
            path = frames_dir / name
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            records.append({'source_bvid': source['bvid'], 'seconds': seconds, 'frame_index': index,
                            'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'size': [int(frame.shape[1]), int(frame.shape[0])],
                            'label': None})
        capture.release()
        print(source['bvid'], 'done', flush=True)
    (OUT / 'frames_manifest.json').write_text(json.dumps({
        'plan': plan['plan'],
        'stage': 'frames_frozen_labels_pending',
        'extracted_at': __import__('time').strftime('%Y-%m-%dT%H:%M:%S'),
        'model_inference_performed': False,
        'frames': records,
        'rule': 'labels must be recorded before any model sees these frames',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('frames', len(records), '->', OUT)


if __name__ == '__main__':
    main()
