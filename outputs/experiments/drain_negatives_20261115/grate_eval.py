"""Frozen whole-frame and floor-grate region diagnostic, evaluated separately."""
import argparse
import json

import cv2
import numpy as np
from ultralytics import YOLO
from train import HERE, PREVIOUS, VIDEO_ROOT, digest, read


def region_match(box, polygon):
    if not len(polygon):
        return {"center_inside":False, "intersects":False, "iou":None}
    x1, y1, x2, y2 = box
    candidate = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    overlap, _ = cv2.intersectConvexConvex(candidate, polygon)
    union = cv2.contourArea(polygon) + cv2.contourArea(candidate) - overlap
    center = ((x1+x2)/2,(y1+y2)/2)
    return {"center_inside":cv2.pointPolygonTest(polygon,center,False)>=0,
            "intersects":overlap>0, "iou":overlap/union if union>0 else 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arms", nargs="+", choices=["facility_42","control_42","facility_73","control_73"])
    args = p.parse_args()
    labels = read(HERE / "grate_labels.json")
    source = labels["source"]
    assert source not in (PREVIOUS / "train.txt").read_text()
    assert source not in {r["source"] for r in read(HERE / "samples.json")}
    video, = VIDEO_ROOT.glob(f"*{source}*.mp4")
    assert digest(video) == labels["video_sha256"]
    scale = np.float32(labels["source_size"]) / np.float32(labels["annotation_size"])
    frames = []
    cap = cv2.VideoCapture(str(video))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == labels["fps"]
        for row in labels["frames"]:
            assert cap.set(cv2.CAP_PROP_POS_FRAMES,row["frame"])
            ok, img = cap.read()
            assert ok and list(img.shape[1::-1]) == labels["source_size"]
            frames.append(img)
    finally:
        cap.release()
    folder = HERE / "evaluation"
    folder.mkdir(exist_ok=True)
    for arm in args.arms:
        weight = HERE / "runs" / arm / "weights/last.pt"
        results = YOLO(str(weight)).predict(frames,imgsz=640,rect=True,device=0,conf=.01,verbose=False)
        rows = []
        for row, result in zip(labels["frames"], results, strict=True):
            polygon = np.float32(row["polygon"]).reshape(-1,2) * scale
            boxes = [{"xyxy":b, "score":s, **region_match(b,polygon)} for b, s in
                     zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())]
            rows.append({"frame":row["frame"],"board":row["board"], "polygon":polygon.tolist(),
                         "highest_score":max((b["score"] for b in boxes),default=None), "boxes":boxes})
        summary = {}
        for cutoff in (.25,.5):
            matched = [[b for b in r["boxes"] if b["score"]>=cutoff] for r in rows]
            summary[str(cutoff)] = {
                "boxes_all":sum(map(len,matched)),
                "fullframe_negatives":sum(r["board"]=="absent" for r in rows),
                "false_frames_confirmed_absent":sum(bool(boxes) and r["board"]=="absent" for boxes,r in zip(matched,rows)),
                "frames_with_grate_region":sum(bool(r["polygon"]) for r in rows),
                "frames_grate_center":sum(any(b["center_inside"] for b in boxes) for boxes in matched),
                "frames_grate_intersection":sum(any(b["intersects"] for b in boxes) for boxes in matched),
            }
        output = {"arm":arm,"weight_sha256":digest(weight),"source":source,"video_sha256":labels["video_sha256"],
                  "score_floor":.01,"summary":summary,"frames":rows}
        path = folder/f"grate_{arm}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(output,indent=2),encoding="utf-8")
        tmp.replace(path)
        print(arm,summary)


if __name__ == "__main__":
    main()
