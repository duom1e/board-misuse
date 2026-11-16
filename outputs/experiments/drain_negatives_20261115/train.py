"""Matched training. Preserve every sampler slot when replacing a repeated negative."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BOARD_OUTPUTS", "D:/CodexProjects/board-misuse/outputs"))
PREVIOUS = ROOT / "experiments/pool4_positive15_seed73_20261115"
CASES = ROOT / "experiments/multisource_paired15_20261111/plan.json"
START = ROOT / "training_runs/board_v2_real/weights/best.pt"
VIDEO_ROOT = ROOT.parent / "待导入视频"
VIDEO_EVAL = {"BV1bY4y1Y7v5","BV1b1ixB5ERA","BV1Go4y1o7ym","BV1264y1F7a6",
              "BV1Mg411U7SV","BV1Zq4y1j7mm"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def label_of(image):
    parts = list(image.parts)
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def prepare(arm):
    import re
    folder = HERE / "data" / arm
    keys = sorted(Path(s) for s in (PREVIOUS / "train.txt").read_text().splitlines())
    valid = sorted(Path(s) for s in (ROOT / "board_dataset/validation.txt").read_text().splitlines())
    assert len(keys) == 270
    cases = read(CASES)["cases"]
    protected = VIDEO_EVAL | {v for c in cases for v in re.findall(r"BV[0-9A-Za-z]{10}", c["image"])}
    eval_hashes = {digest(Path(c["image"])) for c in cases} | {digest(p) for p in valid}
    replaces = {}
    if arm == "facility":
        records = read(HERE / "samples.json")
        assert records and len({r["replace"] for r in records}) == len(records)
        for record in records:
            source = record["source"]
            if source in protected:
                raise ValueError(f"Reserved source {source}")
            slots = [i for i, p in enumerate(keys) if p.name == record["replace"]]
            assert len(slots) == 2 and not label_of(keys[slots[-1]]).read_text().strip()
            replaces[slots[-1]] = record
    evidence = []
    hashes = {}
    for split, paths in (("train",keys), ("val",valid)):
        images = folder / "images" / split
        labels = folder / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for slot, original in enumerate(paths):
            name = f"{slot:04}"
            dst = images / (name + original.suffix.lower())
            label = labels / (name + ".txt")
            if split != "train" or slot not in replaces:
                shutil.copyfile(original, dst)
                shutil.copyfile(label_of(original), label)
                continue
            r = replaces[slot]
            candidates = list(VIDEO_ROOT.glob(f"*{r['source']}*.mp4"))
            assert len(candidates) == 1
            video = candidates[0]
            if video not in hashes:
                hashes[video] = digest(video)
            assert hashes[video] == r["video_sha256"]
            cap = cv2.VideoCapture(str(video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            ok, img = cap.read()
            cap.release()
            assert ok
            if r.get("crop"):
                x1,y1,x2,y2 = r["crop"]
                assert 0 <= x1 < x2 <= img.shape[1] and 0 <= y1 < y2 <= img.shape[0]
                img = img[y1:y2,x1:x2]
            ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            assert ok
            dst.write_bytes(encoded.tobytes())
            label.write_bytes(b"")
            image_hash = digest(dst)
            assert image_hash not in eval_hashes
            evidence.append({**r, "slot":slot, "image_sha256":image_hash})
    if evidence:
        assert len({item["image_sha256"] for item in evidence}) == len(evidence)
        (HERE / "sample_audit.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False),encoding="utf-8")
    config = folder / "data.yaml"
    config.write_text(yaml.safe_dump({"path":folder.as_posix(), "train":"images/train","val":"images/val","names":["cutting_board"]}),encoding="utf-8")
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["control","facility"])
    p.add_argument("seeds", nargs="+", type=int)
    args = p.parse_args()
    config = prepare(args.arm)
    for seed in args.seeds:
        name = f"{args.arm}_{seed}"
        if (HERE / "runs" / name / "results.csv").exists():
            raise FileExistsError(name)
        YOLO(str(START)).train(data=str(config),epochs=15,batch=8,nbs=8,imgsz=640,workers=0,
            device=0,seed=seed,optimizer="AdamW",lr0=.0001,warmup_epochs=0,amp=False,
            mosaic=0,plots=False,project=str(HERE / "runs"),name=name,exist_ok=True)


if __name__ == "__main__":
    main()
