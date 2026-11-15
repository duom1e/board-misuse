from pathlib import Path
import argparse
import hashlib
import json
import cv2
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    root = Path("D:/CodexProjects/board-misuse")
    folder = Path(__file__).resolve().parent / args.source
    folder.mkdir(exist_ok=True)
    video = next((root/"待导入视频").glob(f"*{args.source}*.mp4"))
    run = root/"outputs/experiments/pool4_positive15_seed73_20261115"
    assert args.source not in (run/"train.txt").read_text()
    weight = run/"run/weights/last.pt"
    model = YOLO(str(weight))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start = round(args.start*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    limit = round(args.seconds*fps) if args.seconds else None
    rows = []
    device = 0 if torch.cuda.is_available() else "cpu"
    while limit is None or len(rows)<limit:
        batch = []
        for _ in range(8 if limit is None else min(8,limit-len(rows))):
            ok, frame = cap.read()
            if not ok: break
            batch.append(frame)
        if not batch: break
        for r in model.predict(batch,imgsz=640,rect=True,conf=.25,device=device,verbose=False):
            rows.append([dict(confidence=s,xyxy=b) for s,b in zip(r.boxes.conf.cpu().tolist(),r.boxes.xyxy.cpu().tolist())])
    assert rows and (limit is None or len(rows)==limit)
    meta=dict(source=args.source,video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),fps=fps,start_frame=start,frames=len(rows),container_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),confidence=.25,window=5)
    cap.release()
    (folder/'meta.json').write_text(json.dumps(meta,indent=2))
    (folder/'positive_frames.json').write_text(json.dumps(rows))
    print(meta)
if __name__=='__main__':main()
