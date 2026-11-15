"""从食堂巡检片段里找"排水沟/地砖"持续误报帧，作为训练负例候选。

用 pool4 补图种子的 last.pt，在 BV1bY4y1Y7v5 的 880–920 秒段上逐帧推理（640 输入、conf 0.25，
与 temporal_transfer_v2 相同条件），把有候选的连续帧合并成段，每段取最高分帧，
导出到 images/ 并写 manifest（含时间、分数、框）。
随后需**人工核对**这些帧确实没有砧板，才能当负例。
"""
import json
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'drain_negative_20261115'
WEIGHTS = BASE / 'experiments' / 'pool4_positive15_seed73_20261115' / 'run' / 'weights' / 'last.pt'
VIDEO = BASE.parent / '待导入视频' / '学校食堂食品安全如何？延安市场监管带你看小学食堂 [BV1bY4y1Y7v5].mp4'
START_S, SECONDS = 880, 40
CONF = 0.25


def main():
    from ultralytics import YOLO

    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(START_S * fps))
    total = int(SECONDS * fps)
    hits = []
    for i in range(total):
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0]
        boxes = result.boxes
        if boxes is not None and len(boxes):
            best = max(range(len(boxes)), key=lambda k: float(boxes.conf[k]))
            hits.append({'index': i, 'seconds': round(START_S + i / fps, 2),
                         'score': round(float(boxes.conf[best]), 4),
                         'box': [round(float(v), 1) for v in boxes.xyxy[best].tolist()]})
    capture.release()

    # 合并连续帧为段
    segments = []
    for hit in hits:
        if segments and hit['index'] - segments[-1][-1]['index'] <= 2:
            segments[-1].append(hit)
        else:
            segments.append([hit])
    peaks = []
    for number, segment in enumerate(segments):
        peak = max(segment, key=lambda h: h['score'])
        peaks.append({'segment': number, 'start': segment[0]['seconds'], 'end': segment[-1]['seconds'],
                      'frames': len(segment), 'peak': peak})
    (OUT / 'manifest.json').write_text(json.dumps({
        'video': str(VIDEO), 'segment_seconds': [START_S, START_S + SECONDS],
        'conf': CONF, 'imgsz': 640, 'weights': str(WEIGHTS),
        'hits': len(hits), 'segments': len(segments), 'peaks': peaks,
        'next': '人工核对每段峰值帧是否真的没有砧板；确认后才可作训练负例',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    # 导出峰值帧图片
    capture = cv2.VideoCapture(str(VIDEO))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(START_S * fps))
    wanted = {peak['peak']['index']: peak for peak in peaks}
    saved = []
    for i in range(total):
        ok, frame = capture.read()
        if not ok:
            break
        if i in wanted:
            path = OUT / 'images' / ('peak_%03d_%05d.jpg' % (wanted[i]['segment'], i))
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            saved.append(str(path))
    capture.release()
    print(json.dumps({'hits': len(hits), 'segments': len(segments), 'saved': len(saved)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
