"""Train matched 285-slot arms and measure presence on the previously viewed kitchen videos."""
import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BOARD_OUTPUTS", "D:/CodexProjects/board-misuse/outputs"))
DATA = ROOT / "experiments/rerun_yolo11n_kitchenneg_seed11_20261115/train.txt"
HOLDOUT = ROOT / "experiments/blind_test_v1"
DEV = ROOT / "experiments/multisource_paired15_20261111/plan.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(path):
    parts = list(path.parts)
    parts[len(parts)-1-parts[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def train(arm, seed):
    ordered = sorted(Path(s) for s in DATA.read_text().splitlines() if s.strip())
    valid = sorted(Path(s) for s in (ROOT/"board_dataset/validation.txt").read_text().splitlines())
    duplicates = Counter(ordered)
    eligible = [p for p in sorted(duplicates) if duplicates[p] == 2 and not label(p).read_text().strip()]
    assert len(ordered) == 285 and len(eligible) >= 8
    replacements = {}
    if arm == "food":
        foods = sorted((HERE/"images").glob("*.jpg"))
        assert 3 <= len(foods) <= len(eligible)
        forbidden = set(load(HOLDOUT/"labels_blind.json")["sources"]) | {"BV1b1ixB5ERA","BV1bY4y1Y7v5","BV1264y1F7a6","BV1Go4y1o7ym"}
        cases = load(DEV)["cases"]
        for case in cases:
            forbidden.update(re.findall(r"BV[0-9A-Za-z]{10}",case["image"]))
        held_hashes = {sha(Path(c["image"])) for c in cases} | {sha(p) for p in (HOLDOUT/"frames").glob("*.jpg")} | {sha(p) for p in valid}
        assert len({sha(p) for p in foods}) == len(foods)
        for p, old in zip(foods,eligible):
            ids = re.findall(r"BV[0-9A-Za-z]{10}",p.name)
            assert len(ids)==1 and ids[0] not in forbidden and sha(p) not in held_hashes
            assert not label(p).read_text().strip()
            slot = max(i for i,v in enumerate(ordered) if v == old)
            ordered[slot] = p
            replacements[slot] = {"previous":old.name,"new":p.name,"sha256":sha(p)}
    base = HERE/"data"/arm
    for split, paths in (("train",ordered),("val",valid)):
        for sub in ("images","labels"):
            (base/sub/split).mkdir(parents=True,exist_ok=True)
        for i,p in enumerate(paths):
            shutil.copyfile(p,base/"images"/split/f"{i:04}{p.suffix.lower()}")
            shutil.copyfile(label(p),base/"labels"/split/f"{i:04}.txt")
    config = base/"data.yaml"
    config.write_text(yaml.safe_dump({"path":base.as_posix(),"train":"images/train","val":"images/val","names":["cutting_board"]}))
    name = f"{arm}_{seed}"
    if (HERE/"runs"/name/"results.csv").exists():
        raise FileExistsError(name)
    initial = ROOT/"yolo11n.pt"
    (HERE/f"plan_{name}.json").write_text(json.dumps({"seed":seed,"entries":len(ordered),"replacements":replacements,"start_sha256":sha(initial)},indent=2))
    YOLO(str(initial)).train(data=str(config),epochs=15,batch=8,nbs=8,imgsz=640,workers=0,
        device=0,seed=seed,optimizer="AdamW",lr0=.0001,warmup_epochs=0,amp=False,mosaic=0,
        plots=False,project=str(HERE/"runs"),name=name,exist_ok=True)


def iou(box, ref):
    x1,y1,x2,y2 = box
    a,b,c,d = ref
    common = max(0,min(x2,c)-max(a,x1))*max(0,min(y2,d)-max(b,y1))
    area = (x2-x1)*(y2-y1)+(c-a)*(d-b)-common
    return common/area if area else 0


def evaluate(name, weights):
    model = YOLO(str(weights))
    observations = []
    for filename, r in load(HOLDOUT/"labels_blind.json")["labels"].items():
        observations.append(("kitchens",HOLDOUT/"frames"/filename,r["board_present"],None))
    for r in load(DEV)["cases"]:
        observations.append(("dev",Path(r["image"]),"yes" if r["gt"] else "no",r["gt"]))
    for group, folder in (("white","white_round_holdout_20261115"),("chicken","chicken_block_full12_20261115")):
        for r in load(ROOT/"experiments"/folder/"annotations.json"):
            observations.append((group,Path(r["image"]),"yes",r["box"]))
    rows = []
    for group, path, presence, gt in observations:
        out = model.predict(str(path),imgsz=640,rect=True,conf=.01,device=0,verbose=False)[0]
        height,width = out.orig_shape
        boxes = [{"xyxy":b,"score":s,"area":(b[2]-b[0])*(b[3]-b[1])/(width*height),"iou":iou(b,gt) if gt else None}
            for b,s in zip(out.boxes.xyxy.cpu().tolist(),out.boxes.conf.cpu().tolist())]
        rows.append({"group":group,"name":path.name,"presence":presence,"boxes":boxes})
    summary = []
    for t in (.25,.5):
        for group in ("kitchens","dev","white","chicken"):
            sub = [r for r in rows if r["group"]==group and r["presence"] in {"yes","no"}]
            pos = [r for r in sub if r["presence"]=="yes"]
            neg = [r for r in sub if r["presence"]=="no"]
            hits = sum(any(b["score"]>=t and (b["iou"] is None or b["iou"]>=.5) for b in r["boxes"]) for r in pos)
            fp = sum(any(b["score"]>=t for b in r["boxes"]) for r in neg)
            summary.append({"threshold":t,"group":group,"hits":hits,"positives":len(pos),"false_frames":fp,"negatives":len(neg)})
    (HERE/f"eval_{name}.json").write_text(json.dumps({"weight_sha256":sha(weights),"rows":rows,"summary":summary},indent=2),encoding="utf-8")
    print(name,summary,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["train","eval"])
    parser.add_argument("arm",choices=["control","food","previous"])
    parser.add_argument("seed",type=int,default=11,nargs="?")
    args = parser.parse_args()
    if args.action == "train":
        assert args.arm != "previous"
        train(args.arm,args.seed)
    else:
        name = f"{args.arm}_{args.seed}"
        weight = ROOT/"training_runs/board_neg15_v1/weights/last.pt" if args.arm=="previous" else HERE/"runs"/name/"weights/last.pt"
        evaluate(name,weight)
