"""Screen a fixed grid before predictions and score the preselected no-board frames."""
import argparse
import json
import re
from pathlib import Path

import cv2
from PIL import Image, ImageDraw
from train_compare import HERE, ROOT, DATA, DEV, HOLDOUT, load, sha

SOURCES = ["BV16g9nYNE6d","BV1tP4y1R7Ye","BV1dK4y1M7Qe","BV1eB4y1J7B6","BV1PW411F7mD",
           "BV15T41117hY","BV1ax7pz1E4v","BV1kp42127BE","BV1XUSbYiEUp","BV1yz4y1d7iF"]
FRACTIONS = (.03,.19,.35,.51,.67,.83,.97)


def frame(cap, number):
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,number)
    ok, bgr = cap.read()
    assert ok and round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == number+1
    return bgr


def video(source):
    paths = list((ROOT.parent/"待导入视频").glob(f"*{source}*.mp4"))
    assert len(paths)==1
    return paths[0]


def screen():
    reserved = set(re.findall(r"BV[0-9A-Za-z]{10}", DATA.read_text()+str(load(DEV)["cases"])))
    reserved.update(load(HOLDOUT/"labels_blind.json")["sources"])
    reserved.update(r["source"] for r in load(HERE/"samples.json"))
    reserved.update(("BV1bY4y1Y7v5","BV1b1ixB5ERA","BV1Go4y1o7ym","BV1264y1F7a6"))
    assert not reserved.intersection(SOURCES)
    for group in range(2):
        sheet = Image.new("RGB",(7*177,5*122),"white")
        draw = ImageDraw.Draw(sheet)
        for row, source in enumerate(SOURCES[group*5:group*5+5]):
            cap = cv2.VideoCapture(str(video(source)))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for col, fraction in enumerate(FRACTIONS):
                number = int(count*fraction)
                im = Image.fromarray(cv2.cvtColor(frame(cap,number),cv2.COLOR_BGR2RGB))
                im.thumbnail((175,98))
                x,y = col*177,row*122
                sheet.paste(im,(x,y))
                draw.text((x,y+99),f"{source} {number}",fill="black")
            cap.release()
        sheet.save(HERE/f"dish_screen_{group}.jpg",quality=90)


def run():
    from ultralytics import YOLO
    labels = load(HERE/"dish_labels.json")
    results = {}
    for name in ("previous_11","control_11","food_11"):
        weights = ROOT/"training_runs/board_neg15_v1/weights/last.pt" if name=="previous_11" else HERE/"runs"/name/"weights/last.pt"
        model = YOLO(str(weights))
        rows = []
        for item in labels:
            path = video(item["source"])
            assert sha(path)==item["video_sha256"]
            cap = cv2.VideoCapture(str(path))
            bgr = frame(cap,item["frame"])
            cap.release()
            if item.get("crop_fraction"):
                h,w = bgr.shape[:2]
                x1,y1,x2,y2 = item["crop_fraction"]
                bgr = bgr[round(y1*h):round(y2*h), round(x1*w):round(x2*w)]
            boxes = model.predict(bgr,imgsz=640,rect=True,conf=.01,device=0,verbose=False)[0].boxes
            rows.append({**item,"boxes":[{"xyxy":b,"score":s} for b,s in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist())]})
        results[name] = {"weight_sha256":sha(weights),"rows":rows,"false_frames":{
            str(t):sum(any(b["score"] >= t for b in r["boxes"]) for r in rows) for t in (.25,.5)}}
    (HERE/"dish_results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("screen","run"))
    args=p.parse_args()
    (screen if args.action=="screen" else run)()
