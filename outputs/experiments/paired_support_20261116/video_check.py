"""Replay the frozen 40-second canteen segment for the two seed-matched pairs."""
import importlib.util
import json

import cv2
from ultralytics import YOLO

from evaluate import weight
from train import EXP, HERE, ROOT, read, sha

SOURCE = "BV1bY4y1Y7v5"
ARMS = ("food_11", "paired_11", "food_42", "paired_42")


def predictions(video, metadata, model):
    cap = cv2.VideoCapture(str(video))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == metadata["fps"]
        assert cap.set(cv2.CAP_PROP_POS_FRAMES, metadata["start_frame"])
        rows = []
        while len(rows) < metadata["frames"]:
            batch = []
            for _ in range(min(8, metadata["frames"] - len(rows))):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Decode stopped at frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
                batch.append(frame)
            out = model.predict(batch, imgsz=640, rect=True, conf=.25, device=0, verbose=False)
            assert len(out) == len(batch)
            for r in out:
                rows.append([dict(confidence=s, xyxy=b) for s,b in
                             zip(r.boxes.conf.cpu().tolist(), r.boxes.xyxy.cpu().tolist())])
        assert round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == metadata["start_frame"] + len(rows)
        return rows
    finally:
        cap.release()


def main():
    previous = EXP / "temporal_transfer_v2"
    spec = importlib.util.spec_from_file_location("transfer_score", previous / "score.py")
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    assert scoring.CONFIDENCE == .25 and scoring.IOU == .5
    original = read(previous / SOURCE / "meta.json")
    paths = list((ROOT.parent / "待导入视频").glob(f"*{SOURCE}*.mp4"))
    assert len(paths) == 1 and sha(paths[0]) == original["video_sha256"]
    assert (original["start_frame"], original["frames"], original["fps"]) == (26400,1200,30)
    meta = {k:original[k] for k in ("source","video_sha256","fps","start_frame","frames")}
    result = {"meta":meta, "confidence":.25, "iou":.5, "imgsz":640, "rect":True, "arms":{}}
    for arm in ARMS:
        path = weight(arm)
        frames = predictions(paths[0], meta, YOLO(str(path)))
        windows = {str(w):scoring.score(frames,meta,{},w) for w in (1,5)}
        result["arms"][arm] = {"weight_sha256":sha(path),"windows":windows}
        print(arm, [(w, row["episode_count"],row["flagged_frames"]) for w,row in windows.items()], flush=True)
    output = HERE / "video_results.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
