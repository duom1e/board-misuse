"""Audit matched training artifacts and summarize the locked development evaluations."""
import csv
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS = ("control_42", "facility_42", "control_73", "facility_73")
NEGATIVES = {"canteen": "BV1bY4y1Y7v5", "stove": "BV1b1ixB5ERA"}
POSITIVES = {"white": "BV1Go4y1o7ym", "chicken": "BV1264y1F7a6"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def audit_data():
    replacements = {entry["slot"]: entry for entry in read(HERE / "sample_audit.json")}
    assert len(replacements) == 4
    material = HERE / "data"
    differences = []
    totals = {}
    for split in ("train", "val"):
        control = material / "control" / "images" / split
        facility = material / "facility" / "images" / split
        names = sorted(p.name for p in control.iterdir() if p.is_file())
        assert names == sorted(p.name for p in facility.iterdir() if p.is_file())
        assert names and (split != "train" or len(names) == 270)
        totals[split] = len(names)
        for name in names:
            a = material / "control" / "labels" / split / Path(name).with_suffix(".txt")
            b = material / "facility" / "labels" / split / Path(name).with_suffix(".txt")
            assert a.read_bytes() == b.read_bytes(), (split, name, "label mismatch")
            current_hash = sha(facility / name)
            if current_hash != sha(control / name):
                slot = int(Path(name).stem)
                assert split == "train" and slot in replacements and not b.read_text().strip()
                assert current_hash == replacements[slot]["image_sha256"]
                differences.append(slot)
    assert set(differences) == set(replacements)
    return {"train_slots": totals["train"], "validation_slots": totals["val"],
            "changed_train_slots": differences, "unchanged_train_images": totals["train"] - len(differences),
            "label_differences": 0, "validation_image_differences": 0}


def summarize():
    audit = audit_data()
    result = {"data_audit": audit, "arms": {}}
    case_hashes = None
    training_settings = None
    video_hashes = None
    for arm in RUNS:
        folder = HERE / "runs" / arm
        with (folder / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            epochs = [int(r["epoch"]) for r in csv.DictReader(stream)]
        assert epochs == list(range(1, 16)), (arm, epochs)
        args = yaml.safe_load((folder / "args.yaml").read_text(encoding="utf-8-sig"))
        assert args["seed"] == int(arm.rsplit("_",1)[1])
        shared = {k:v for k,v in args.items() if k not in {"name", "save_dir", "data", "seed"}}
        if training_settings is None:
            training_settings = shared
        assert shared == training_settings
        record = read(HERE / "evaluation" / f"{arm}.json")
        weight_hash = sha(folder / "weights/last.pt")
        assert record["arm"] == arm and record["weight_sha256"] == weight_hash
        assert record["imgsz"] == 640 and record["video_confidence"] == .25 and record["iou"] == .5
        bindings = [(r["image_sha256"],r["gt"]) for r in record["static"]["cases"]]
        videos = {k:v["meta"]["video_sha256"] for k,v in record["videos"].items()}
        if case_hashes is None:
            case_hashes, video_hashes = bindings, videos
        assert bindings == case_hashes and videos == video_hashes and len(bindings) == 84
        summary = {"epochs": len(epochs), "weight_sha256": weight_hash,
                   "static": record["static"]["summary"], "videos": {}}
        for label, source in {**NEGATIVES, **POSITIVES}.items():
            data = record["videos"][source]
            expected = {"canteen":1200, "stove":3503, "white":60, "chicken":60}[label]
            assert data["frames_evaluated"] == expected
            series = {}
            for window in ("1", "5"):
                values = data["windows"][window]
                if label in NEGATIVES:
                    series[window] = {"events":values["episode_count"], "flagged_frames":values["flagged_frames"]}
                else:
                    assert len(values["anchors"]) == 12
                    series[window] = {"hits":values["anchor_hits"], "anchors":12}
            summary["videos"][label] = series
        result["arms"][arm] = summary
    result["video_sha256"] = video_hashes
    result["training"] = {k:training_settings[k] for k in ("epochs","batch","imgsz","optimizer","lr0","nbs","warmup_epochs","amp","mosaic")}
    return result


def render(result):
    lines = ["# 厨房设施负例的双种子案板定位对照", "",
             "同一案板初始权重，每组训练 15 轮，每轮 270 条输入、34 批。种子 42 和 73 分别重训对照组和设施组，共训练 60 轮，统一使用最后一轮权重。每个种子内两组的训练参数一致。",
             "", "设施组将炒锅和木台面各两张的第二份重复图替换为三段视频中的四张无板图。四张分别为花纹地砖、白色水槽局部、金属水槽及条孔沥水篮。可见的材质和纹理与误报设施近似，训练图中没有地面排水格栅。逐文件核对确认其余 266 个训练位置的图像完全一致，所有标签和验证图片也一致。",
             "", "## 固定图片开发集", "",
             "固定 84 张图片内有 14 张有板、70 张无板。有板图按预测框与参考框 IoU 至少 0.5 计命中。无板图只要出现超过或等于当前置信度门槛的框，即记为一张误报。最后一列只统计在本轮已知训练清单中未找到同一 BV 视频来源的负例。", "",
             "| 种子 | 训练组 | 置信度 | 有板命中 | 无板误报 | 排除同源后的无板误报 |",
             "|---:|---|---:|---:|---:|---:|"]
    for arm, data in result["arms"].items():
        seed = arm.rsplit("_",1)[1]
        name = "对照" if arm.startswith("control") else "设施负例"
        for conf in ("0.25", "0.5"):
            all_cases = data["static"]["all"][conf]
            disjoint = data["static"]["known_source_disjoint"][conf]
            assert all_cases["positive_total"] == 14 and all_cases["negative_total"] == 70
            lines.append(f"| {seed} | {name} | {conf} | {all_cases['hits']}/14 | {all_cases['false_frames']}/70 | {disjoint['false_frames']}/{disjoint['negative_total']} |")
    lines.extend(["", "## 视频复测", "",
                  "食堂巡检取 880 至 920 秒，共 1200 帧。炒锅线取全段 3503 帧。窗口 1 为单帧，窗口 5 要求当前候选与最近 5 帧中每帧至少一个候选的 IoU 达到 0.5。两者均采用置信度 0.25。连续有候选的帧合并为一段，同一物体可被计入多段。事件统计记录提示数量，具体所指物体需查看各段峰值。",
                  "", "| 种子 | 训练组 | 窗口 | 食堂段数 | 食堂有框帧 | 炒锅段数 | 炒锅有框帧 | 白圆板命中 | 切鸡圆板命中 |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
    for arm, data in result["arms"].items():
        seed = arm.rsplit("_",1)[1]
        name = "对照" if arm.startswith("control") else "设施负例"
        for w in ("1","5"):
            v = {k:series[w] for k,series in data["videos"].items()}
            lines.append(f"| {seed} | {name} | {w} | {v['canteen']['events']} | {v['canteen']['flagged_frames']} | {v['stove']['events']} | {v['stove']['flagged_frames']} | {v['white']['hits']}/12 | {v['chicken']['hits']}/12 |")
    lines.extend(["", "两个圆板视频各沿用 12 个定位锚点，每个锚点读取当帧与之前四帧。部分参考框按可见板面近似绘制。本次开发复测衡量案板位置检出；案板用途、食材接触和错用事件仍需单独标注评估。"])
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    results = summarize()
    report = render(results)
    (HERE / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False)+"\n",encoding="utf-8")
    (HERE / "报告.md").write_text(report,encoding="utf-8")
    print(report)
