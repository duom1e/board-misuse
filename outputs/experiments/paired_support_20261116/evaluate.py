"""Evaluate board presence and box geometry separately on the reviewed inputs."""
import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from train import EXP, HERE, PREV, ROOT, read, read_frame, sha

ARMS = ("original","food_11","paired_11","food_42","paired_42")


def weight(arm):
    if arm=="original":
        return ROOT/"training_runs/board_neg15_v1/weights/last.pt"
    base = PREV if arm=="food_11" else HERE
    return base/"runs"/arm/"weights/last.pt"


def iou(box,ref):
    x,y,u,v=box
    a,b,c,d=ref
    inter=max(0,min(u,c)-max(x,a))*max(0,min(v,d)-max(y,b))
    return inter/((u-x)*(v-y)+(c-a)*(d-b)-inter)


def inputs():
    records=[]
    def add(group,name,image,presence,gt=None,source=None,old=None):
        if isinstance(image,Path):
            image=cv2.imread(str(image))
        assert image is not None
        records.append((dict(group=group,name=name,presence=presence,gt=gt,source=source,old=old),image))

    hist=EXP/"blind_test_v1"
    reviewed=read(HERE/"geometry/labels.json")+read(HERE/"geometry/additional_labels.json")
    by_name={r["name"]:r for r in reviewed}
    assert len(by_name)==len(reviewed)
    for name,r in read(hist/"labels_blind.json")["labels"].items():
        path=hist/"frames"/name
        old=r["board_present"]
        geo=by_name.get(name,{})
        if geo.get("image_sha256"):
            assert geo["image_sha256"]==sha(path)
        add("kitchens",name,path,geo.get("presence_review",old),geo.get("box"),name[:12],old)
    for row in read(EXP/"multisource_paired15_20261111/plan.json")["cases"]:
        path=Path(row["image"])
        add("dev",path.name,path,"yes" if row["gt"] else "no",row["gt"])
    for group,folder in (("white","white_round_holdout_20261115"),("chicken","chicken_block_full12_20261115")):
        for row in read(EXP/folder/"annotations.json"):
            path=Path(row["image"])
            ref=row["box"]
            if group=="chicken":
                ref={"00":[52,0,1216,720],"07":[59,0,1280,720]}.get(path.stem[-2:],ref)
            add(group,path.name,path,"yes",ref)
    for row in read(PREV/"dish_labels.json"):
        image=read_frame(row)
        if row.get("crop_fraction"):
            x,y,u,v=row["crop_fraction"]
            h,w=image.shape[:2]
            image=image[round(y*h):round(v*h),round(x*w):round(u*w)]
        add("dishes",str(row["frame"]),image,"no",source=row["source"])
    for row in read(HERE/"transfer/labels.json"):
        gt=row["box"]
        add("transfer",str(row["frame"]),read_frame(row),"yes" if gt else "no",gt,row["source"])
    positives=read(HERE/"positive/samples.json")
    for row in positives:
        add("fit",str(row["frame"]),read_frame(row),"yes",row["box"],row["source"])
    for row in read(PREV/"prepared.json"):
        path=PREV/"images"/row["image"]
        assert sha(path)==row["sha256"]
        add("fit",path.name,path,"no",source=path.name[:12])
    return records


def summarize(rows,threshold):
    summary={}
    for group in sorted({r["group"] for r in rows}):
        part=[r for r in rows if r["group"]==group]
        present=[r for r in part if r["presence"]=="yes"]
        absent=[r for r in part if r["presence"]=="no"]
        boxed=[r for r in present if r["gt"]]
        def kept(r):
            return [b for b in r["boxes"] if b["score"]>=threshold]
        def location(r):
            return any(b["iou"]>=.5 for b in kept(r))
        metrics={"presence_hits":sum(bool(kept(r)) for r in present),"positives":len(present),
                 "located":sum(location(r) for r in boxed),"boxed":len(boxed),
                 "false_frames":sum(bool(kept(r)) for r in absent),"negatives":len(absent)}
        if group=="transfer":
            metrics["paired_source_pass"]={s:all(location(r) if r["gt"] else not kept(r)
                for r in part if r["source"]==s) for s in sorted({r["source"] for r in part})}
        if group=="kitchens":
            metrics["old_presence_hits"]=sum(bool(kept(r)) for r in part if r["old"]=="yes")
            metrics["old_false_frames"]=sum(bool(kept(r)) for r in part if r["old"]=="no")
        summary[group]=metrics
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("arms",nargs="*",default=list(ARMS))
    args=p.parse_args()
    records=inputs()
    folder=HERE/"evaluation"
    folder.mkdir(exist_ok=True)
    for arm in args.arms:
        assert arm in ARMS
        path=weight(arm)
        model=YOLO(str(path))
        rows=[]
        for label,image in records:
            pred=model.predict(image,imgsz=640,rect=True,conf=.01,device=0,verbose=False)[0].boxes
            h,w=image.shape[:2]
            boxes=[dict(score=s,xyxy=b,area=(b[2]-b[0])*(b[3]-b[1])/(h*w),
                        iou=iou(b,label["gt"]) if label["gt"] else None)
                   for s,b in zip(pred.conf.cpu().tolist(),pred.xyxy.cpu().tolist())]
            rows.append({**label,"boxes":boxes})
        counts={str(t):summarize(rows,t) for t in (.25,.5)}
        (folder/f"{arm}.json").write_text(json.dumps({"sha256":sha(path),"rows":rows,"summary":counts},
                ensure_ascii=False,indent=2),encoding="utf-8")
        print(arm,counts["0.25"],flush=True)


if __name__=="__main__":
    main()
