"""从**另一段**食堂视频里找排水/地砖误报帧，避免"在评估片段上训练"。

上一轮用 BV1bY4y1Y7v5 自己的 880–920s 帧当负例，等于在该片段上训练，评估结果无效。
这里改从 BV18pRhBgEyw（千人园区食堂，另一来源）取样，找到同类误报帧后人工核对。
"""
import json
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'drain_negative_src2_20261115'
WEIGHTS = BASE / 'experiments' / 'pool4_positive15_seed73_20261115' / 'run' / 'weights' / 'last.pt'
VIDEO = next((BASE.parent / '待导入视频').glob('*BV18pRhBgEyw*.mp4'))
STEP_SECONDS = 5
CONF = 0.25


def main():
    from ultralytics import YOLO

    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    hits = []
    index = 0
    while index * STEP_SECONDS < duration:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index * STEP_SECONDS * fps))
        ok, frame = capture.read()
        if not ok:
            break
        boxes = model.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0].boxes
        if boxes is not None and len(boxes):
            best = max(range(len(boxes)), key=lambda k: float(boxes.conf[k]))
            hits.append({'seconds': index * STEP_SECONDS, 'score': round(float(boxes.conf[best]), 4),
                         'box': [round(float(v), 1) for v in boxes.xyxy[best].tolist()]})
        index += 1
    capture.release()

    (OUT / 'manifest.json').write_text(json.dumps({
        'video': str(VIDEO), 'sample_step_seconds': STEP_SECONDS, 'conf': CONF,
        'hits': len(hits), 'candidates': hits,
        'next': '逐帧人工核对是否为排水/地砖/无关物体，确认无板后才可作负例',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'hits': len(hits), 'sampled': index}, ensure_ascii=False))


if __name__ == '__main__':
    main()
