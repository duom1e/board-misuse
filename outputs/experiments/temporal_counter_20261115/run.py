from pathlib import Path
import hashlib
import json
import os
import statistics

import cv2
import torch
from ultralytics import YOLO

out = Path(__file__).resolve().parent
base = Path('D:/CodexProjects/board-misuse')
source = next((base/'待导入视频').glob('*BV1b1ixB5ERA*.mp4'))
device = 0 if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2)
cv2.setNumThreads(1)

def durations(scores, threshold, fps):
    events = []
    start = None
    for idx, score in enumerate(scores + [0.0]):
        if score >= threshold and start is None:
            start = idx
        if score < threshold and start is not None:
            events.append({'first_frame':start, 'last_frame':idx-1,
                           'seconds':(idx-start)/fps})
            start = None
    return events

def run(arm):
    run_dir = base/f'outputs/experiments/pool4_{arm}15_seed73_20261115'
    assert 'BV1b1ixB5ERA' not in (run_dir/'train.txt').read_text()
    weight = run_dir/'run/weights/last.pt'
    model = YOLO(str(weight))
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    while True:
        batch = []
        for _ in range(8):
            success, frame = cap.read()
            if not success: break
            batch.append(frame)
        if not batch: break
        for result in model.predict(batch, imgsz=640, rect=True, conf=.1, device=device, verbose=False):
            boxes = [{'confidence':float(score),'xyxy':box}
                     for score,box in zip(result.boxes.conf.cpu().tolist(),result.boxes.xyxy.cpu().tolist())]
            rows.append(boxes)
    cap.release()
    assert len(rows)==count
    scores = [max((box['confidence'] for box in boxes), default=0.) for boxes in rows]
    summary = []
    for threshold in [.5,.25,.1]:
        events = durations(scores,threshold,fps)
        summary.append({'threshold':threshold,'frames':sum(s>=threshold for s in scores),'episodes':len(events),
                        'episodes_at_least_half_second':sum(e['seconds']>=.5 for e in events),
                        'longest_seconds':max((e['seconds'] for e in events),default=0),'events':events})
    (out/f'{arm}_frames.json').write_text(json.dumps(rows))
    data={'source':source.name,'video_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'weight_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'fps':fps,'frames':count,
          'duration_seconds':count/fps,'device':str(device),'summary':summary}
    (out/f'{arm}_summary.json').write_text(json.dumps(data,indent=2))
    print(arm, [{k:v for k,v in s.items() if k!='events'} for s in summary],flush=True)

if __name__=='__main__':
    for arm in ['positive','control']:
        run(arm)
