"""Swap repeated positive slots for reviewed board boxes; keep the five dish negatives and budget fixed."""
import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BOARD_OUTPUTS", "D:/CodexProjects/board-misuse/outputs"))
EXP = ROOT / "experiments"
PREV = EXP / "food_closeup_20261116"
BASE = PREV / "data/food"
INITIAL = ROOT / "yolo11n.pt"


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read_frame(row):
    videos = list((ROOT.parent/"待导入视频").glob(f"*{row['source']}*.mp4"))
    assert len(videos)==1 and sha(videos[0])==row["video_sha256"]
    cap = cv2.VideoCapture(str(videos[0]))
    try:
        assert cap.set(cv2.CAP_PROP_POS_FRAMES,row["frame"])
        ok,img=cap.read()
        assert ok and round(cap.get(cv2.CAP_PROP_POS_FRAMES))==row["frame"]+1
        return img
    finally:
        cap.release()


def prepare(arm):
    out=HERE/"data"/arm
    for kind in ("images","labels"):
        for split in ("train","val"):
            shutil.copytree(BASE/kind/split,out/kind/split,dirs_exist_ok=True)
    ordered=sorted(Path(s) for s in (EXP/"rerun_yolo11n_kitchenneg_seed11_20261115/train.txt").read_text().splitlines())
    count=Counter(ordered)
    assert len(ordered)==285
    slots=[i for i,p in enumerate(ordered) if count[p]>1 and i==max(j for j,v in enumerate(ordered) if v==p)
           and (out/f"labels/train/{i:04}.txt").read_text().strip()]
    old=read(PREV/"plan_food_11.json")
    for slot,r in old["replacements"].items():
        assert sha(out/f"images/train/{int(slot):04}.jpg")==r["sha256"]
    used={}
    if arm=="paired":
        cases=read(EXP/"multisource_paired15_20261111/plan.json")["cases"]
        reserved=set(re.findall(r"BV[0-9A-Za-z]{10}",str(cases))) | set(read(EXP/"blind_test_v1/labels_blind.json")["sources"])
        reserved.update(["BV1bY4y1Y7v5","BV1b1ixB5ERA","BV1Go4y1o7ym","BV1264y1F7a6"])
        reserved.update(r["source"] for r in read(PREV/"dish_labels.json"))
        obs=read(HERE/"positive/samples.json")
        assert len({(r["source"],r["frame"]) for r in obs})==len(obs)>0
        for row in obs:
            source=row["source"]
            assert source not in reserved
            slot=next((i for i in slots if source in ordered[i].name),slots[0])
            slots.remove(slot)
            img=read_frame(row)
            h,w=img.shape[:2]
            a,b,c,d=row["box"]
            assert 0<=a<c<=w and 0<=b<d<=h
            target=out/f"images/train/{slot:04}.jpg"
            assert target.exists() and sha(target)==sha(ordered[slot])
            ok,encoded=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,95])
            assert ok
            target.write_bytes(encoded.tobytes())
            (out/f"labels/train/{slot:04}.txt").write_text(
                f"0 {(a+c)/(2*w):.8f} {(b+d)/(2*h):.8f} {(c-a)/w:.8f} {(d-b)/h:.8f}\n")
            used[slot]={"original":ordered[slot].name,"source":source,"frame":row["frame"],"image_sha256":sha(target)}
    cfg=out/"data.yaml"
    cfg.write_text(yaml.safe_dump({"path":out.as_posix(),"train":"images/train","val":"images/val","names":["cutting_board"]}))
    (HERE/f"plan_{arm}.json").write_text(json.dumps({"entries":285,"init":sha(INITIAL),"replacements":used},indent=2))
    return cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument("arm",choices=("food","paired"))
    p.add_argument("seeds",nargs="+",type=int)
    args=p.parse_args()
    cfg=prepare(args.arm)
    for seed in args.seeds:
        name=f"{args.arm}_{seed}"
        if (HERE/"runs"/name/"results.csv").exists():
            raise FileExistsError(name)
        YOLO(str(INITIAL)).train(data=str(cfg),epochs=15,batch=8,nbs=8,imgsz=640,workers=0,
            device=0,seed=seed,optimizer="AdamW",lr0=.0001,warmup_epochs=0,amp=False,mosaic=0,
            plots=False,project=str(HERE/"runs"),name=name,exist_ok=True)


if __name__=="__main__":
    main()
