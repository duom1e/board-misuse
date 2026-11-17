"""Verify that each seed compares a matched training recipe with five changed positive slots."""
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BOARD_OUTPUTS", "D:/CodexProjects/board-misuse/outputs"))
PREV = ROOT/"experiments/food_closeup_20261116"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_split(dataset, split, count):
    images = {p.stem:p for p in (dataset/"images"/split).iterdir() if p.is_file()}
    labels = {p.stem:p for p in (dataset/"labels"/split).glob("*.txt")}
    assert set(images) == set(labels) == {f"{i:04}" for i in range(count)}
    slots = {}
    for key, image in sorted(images.items()):
        lines = labels[key].read_text().splitlines()
        boxes = []
        for line in lines:
            values = line.split()
            assert len(values)==5 and values[0]=="0"
            box = list(map(float, values[1:]))
            assert all(0 <= x <= 1 for x in box) and box[2] > 0 and box[3] > 0
            boxes.append(box)
        slots[int(key)] = (sha(image),sha(labels[key]),boxes)
    return slots


def datasets():
    sets = {"previous": PREV/"data/food", "food":HERE/"data/food","paired":HERE/"data/paired"}
    plan = read(HERE/"plan_paired.json")
    changes = {int(k):v for k,v in plan["replacements"].items()}
    assert len(changes)==5 and plan["entries"]==285
    source_rows = {(x["source"], x["frame"]):x for x in read(HERE/"positive/samples.json")}
    assert len(source_rows)==5
    report = {}
    for split, count in (("train",285),("val",10)):
        observed = {k:inspect_split(v,split,count) for k,v in sets.items()}
        assert observed["food"] == observed["previous"]
        wanted = set(changes) if split=="train" else set()
        for column in (0,1):
            assert {i for i in range(count) if observed["food"][i][column] != observed["paired"][i][column]} == wanted
        counts = {}
        for arm, slots in observed.items():
            positives = sum(bool(v[2]) for v in slots.values())
            counts[arm] = {"entries":len(slots),"positive_slots":positives,"negative_slots":len(slots)-positives,
                           "board_boxes":sum(len(v[2]) for v in slots.values())}
        assert counts["paired"] == counts["food"] == counts["previous"]
        if split=="train":
            prior_hashes = Counter(slot[0] for slot in observed["food"].values())
            for i, change in changes.items():
                old_image,_,old_boxes = observed["food"][i]
                image,_,boxes = observed["paired"][i]
                assert len(boxes) == len(old_boxes) == 1 and image == change["image_sha256"]
                assert prior_hashes[old_image] >= 2 and any(v[0]==old_image for v in observed["paired"].values())
                assert image not in prior_hashes
                r = source_rows[change["source"],change["frame"]]
                x1,y1,x2,y2 = r["box"]
                w,h = r["width"],r["height"]
                target = [(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h]
                assert all(abs(x-y)<1e-7 for x,y in zip(boxes[0],target))
        report[split] = {"counts":counts,"matched_unchanged_slots":count-len(wanted),"changed_slots":sorted(wanted)}
    return report


def audit():
    data = datasets()
    initial = sha(ROOT/"yolo11n.pt")
    assert initial == read(PREV/"plan_food_11.json")["start_sha256"]
    assert all(initial==read(HERE/f"plan_{arm}.json")["init"] for arm in ("food","paired"))
    paths = {"food_11":PREV/"runs/food_11", **{arm:HERE/"runs"/arm for arm in ("paired_11","food_42","paired_42")}}
    standard = None
    runs = {}
    for arm,path in paths.items():
        args = yaml.safe_load((path/"args.yaml").read_text(encoding="utf-8"))
        seed = int(arm.rsplit("_",1)[-1])
        assert args["seed"]==seed and sha(Path(args["model"]))==initial
        recipe = {k:v for k,v in args.items() if k not in {"name","save_dir","project","data","model","seed"}}
        if standard is None:
            standard = recipe
        assert recipe == standard
        assert recipe["epochs"]==15 and recipe["batch"]==8 and recipe["imgsz"]==640
        with (path/"results.csv").open(encoding="utf-8-sig", newline="") as stream:
            epochs = [int(row["epoch"]) for row in csv.DictReader(stream)]
        assert epochs == list(range(1,16))
        digest = sha(path/"weights/last.pt")
        if arm=="food_11":
            assert read(PREV/"eval_food_11.json")["weight_sha256"] == digest
        runs[arm] = {"seed":seed,"epochs":len(epochs),"weight_sha256":digest,"init_sha256":initial,"reused":arm=="food_11"}
    return {"datasets":data,"runs":runs,"matched_hyperparameters":standard,
            "food_11_reuse":"Previous experiment weight SHA verified against its saved evaluation. All 285 inputs, 285 labels and validation bytes match the current control. Training initialization and all non-path, non-seed parameters match the three new runs."}


if __name__=="__main__":
    result = audit()
    target = HERE/"audit.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(target)
    print(json.dumps(result["datasets"],ensure_ascii=False))
