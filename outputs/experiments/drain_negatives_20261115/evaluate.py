"""Evaluate last-epoch weights against the frozen image cases and four real videos."""
import argparse
import importlib.util
import json
import re
from pathlib import Path

import cv2
from train import CASES, HERE, PREVIOUS, ROOT, VIDEO_ROOT, digest, read

CHICKEN = "BV1264y1F7a6"
WHITE = "BV1Go4y1o7ym"
CANTEEN = "BV1bY4y1Y7v5"
STOVE = "BV1b1ixB5ERA"
EXP = ROOT / "experiments"
spec = importlib.util.spec_from_file_location("frozen_transfer_score", EXP / "temporal_transfer_v2/score.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
legacy.PROJECT = ROOT


def anchors(source, sha):
    if source == CHICKEN:
        return legacy.load_anchors({"video_sha256": sha})
    sources = read(ROOT / "yellow_chicken_sources_v1/sources.json")
    manifest = next(s for s in sources if s["id"] == WHITE)
    assert manifest["sha256"] == sha
    labels = {Path(r["image"]).name: r for r in read(EXP / "white_round_holdout_20261115/annotations.json")}
    result = {}
    for r in manifest["frames"]:
        p = Path(r["image"])
        label = labels[p.name]
        assert digest(p) == r["sha256"] == label["sha"]
        result[r["frame_index"]] = {"label_index": r["index"], "name": p.name, "reference": label["box"]}
    assert len(result) == 12
    return result


def inputs():
    cases = read(CASES)["cases"]
    assert len(cases) == 84 and sum(bool(c["gt"]) for c in cases) == 14
    train_ids = set(re.findall(r"BV[A-Za-z0-9]{10}", (PREVIOUS / "train.txt").read_text()))
    samples = HERE / "samples.json"
    if samples.exists():
        train_ids.update(r["source"] for r in read(samples))
    assert not train_ids.intersection((CHICKEN, WHITE, CANTEEN, STOVE))
    videos = {}
    for source in (CANTEEN, STOVE, CHICKEN, WHITE):
        candidates = list(VIDEO_ROOT.glob(f"*{source}*.mp4"))
        assert len(candidates) == 1
        sha = digest(candidates[0])
        if source in (CANTEEN, STOVE):
            record = (EXP / "temporal_transfer_v2" / CANTEEN / "meta.json" if source == CANTEEN
                      else EXP / "temporal_counter_20261115/positive_summary.json")
            previous = read(record)
            assert previous["video_sha256"] == sha and previous["fps"] == 30
            start = previous.get("start_frame", 0)
            count = previous["frames"]
            assert (start, count) == ((26400, 1200) if source == CANTEEN else (0, 3503))
            labels = {}
        else:
            start, count = None, None
            labels = anchors(source, sha)
        videos[source] = dict(path=candidates[0], sha=sha, start=start, count=count, anchors=labels)
    return cases, train_ids, videos


def predict(model, images, confidence, device):
    results = model.predict(images, imgsz=640, rect=True, conf=confidence, device=device, verbose=False)
    return [[{"xyxy": box, "confidence": score} for box, score in
             zip(row.boxes.xyxy.cpu().tolist(), row.boxes.conf.cpu().tolist())] for row in results]


def static_scores(model, cases, train_ids, device):
    rows = []
    for c in cases:
        image = Path(c["image"])
        source = re.search(r"BV[A-Za-z0-9]{10}", image.name)
        boxes = predict(model, str(image), .01, device)[0]
        if c["gt"]:
            for b in boxes:
                b["iou"] = legacy.iou(b["xyxy"], c["gt"])
        rows.append(dict(image=image.name, image_sha256=digest(image), source=source.group() if source else None,
                         source_in_known_finetune=bool(source and source.group() in train_ids), gt=c["gt"], boxes=boxes))
    summary = {}
    for key, subset in [("all",rows), ("known_source_disjoint", [r for r in rows if r["source"] and not r["source_in_known_finetune"]])]:
        summary[key] = {}
        for threshold in (.25,.5):
            positive = [r for r in subset if r["gt"]]
            negative = [r for r in subset if not r["gt"]]
            summary[key][str(threshold)] = {
                "hits": sum(any(b["confidence"] >= threshold and b["iou"] >= .5 for b in r["boxes"]) for r in positive),
                "positive_total": len(positive), "negative_total": len(negative),
                "false_frames": sum(any(b["confidence"] >= threshold for b in r["boxes"]) for r in negative),
            }
    return {"summary":summary, "cases":rows}


def read_predictions(cap, start, count, model, device):
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
        raise ValueError(f"Cannot seek to {start}")
    rows = []
    for offset in range(0, count, 8):
        batch = []
        for _ in range(min(8, count - offset)):
            ok, img = cap.read()
            if not ok:
                raise ValueError(f"Decode failed at {start + offset + len(batch)}")
            batch.append(img)
        rows.extend(predict(model, batch, .25, device))
    assert len(rows) == count and round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == start + count
    return rows


def video_scores(model, item, device):
    cap = cv2.VideoCapture(str(item["path"]))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps == 30
        meta = {"fps":fps, "video_sha256":item["sha"]}
        if not item["anchors"]:
            meta["start_frame"] = item["start"]
            frames = read_predictions(cap, item["start"], item["count"], model, device)
            return {"meta":meta, "frames_evaluated":len(frames),
                    "windows":{str(w):legacy.score(frames,meta,{},w) for w in (1,5)}}
        details = {1:[],5:[]}
        for number, label in sorted(item["anchors"].items()):
            start = number - 4
            frames = read_predictions(cap, start, 5, model, device)
            for w, rows in details.items():
                result = legacy.score(frames, {**meta, "start_frame":start}, {number:label}, w)
                rows.extend(result["anchors"])
        return {"meta":meta, "frames_evaluated":5 * len(item["anchors"]),
                "windows":{str(w):{"anchors":rows,"anchor_hits":sum(r["hit"] for r in rows)} for w, rows in details.items()}}
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="*")
    parser.add_argument("--device", default="0")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    arms = args.arms or ["control_42", "facility_42", "control_73", "facility_73"]
    if any(a not in {"control_42", "control_73", "facility_42", "facility_73"} for a in arms):
        parser.error("Use control_42, facility_42, control_73, or facility_73.")
    cases, train_ids, videos = inputs()
    if args.check_data:
        print("cases",len(cases), "videos",[(s, i["count"] or len(i["anchors"])) for s,i in videos.items()])
        return
    from ultralytics import YOLO
    folder = HERE / "evaluation"
    folder.mkdir(exist_ok=True)
    for arm in arms:
        weight = HERE / "runs" / arm / "weights/last.pt"
        model = YOLO(str(weight))
        result = {"arm":arm, "weight_sha256":digest(weight), "imgsz":640, "rect":True,
                  "video_confidence":.25, "iou":.5, "static":static_scores(model,cases,train_ids,args.device), "videos":{}}
        for source, item in videos.items():
            result["videos"][source] = video_scores(model, item, args.device)
            print(arm, source, [(w, v.get("anchor_hits") if item["anchors"] else v["episode_count"]) for w,v in result["videos"][source]["windows"].items()], flush=True)
        temp = folder / f"{arm}.json.tmp"
        temp.write_text(json.dumps(result, ensure_ascii=False,indent=2),encoding="utf-8")
        temp.replace(folder / f"{arm}.json")
        print(arm, result["static"]["summary"]["all"],flush=True)


if __name__ == "__main__":
    main()
