"""Verify both training arms and summarize their scores and one previously viewed error."""
import csv
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BOARD_OUTPUTS", "D:/CodexProjects/board-misuse/outputs"))
FRAME = "BV1Mg411U7SV_098.6s.jpg"
NAMES = {"previous_11": "原候选 board_neg15_v1", "control_11": "本轮对照组", "food_11": "食物特写组"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def audited_slots():
    arms = ["control", "food"]
    plans = [load(HERE / f"plan_{arm}_11.json") for arm in arms]
    assert all(p["seed"] == 11 and p["entries"] == 285 for p in plans)
    assert plans[0]["start_sha256"] == plans[1]["start_sha256"] and not plans[0]["replacements"]
    actual = {}
    for split in ["train", "val"]:
        images = [sorted((HERE / "data" / arm / "images" / split).glob("*")) for arm in arms]
        assert {p.name for p in images[0]} == {p.name for p in images[1]} and images[0]
        if split == "train":
            assert len(images[0]) == 285
        for left, right in zip(*images):
            slot = int(left.stem)
            label_paths = [HERE / "data" / arm / "labels" / split / left.with_suffix(".txt").name for arm in arms]
            assert label_paths[0].read_bytes() == label_paths[1].read_bytes()
            if sha(left) != sha(right):
                assert split == "train" and not label_paths[0].read_bytes().strip()
                actual[slot] = sha(right)
    plan = {int(k):v["sha256"] for k,v in plans[1]["replacements"].items()}
    assert len(plan) == 5 and actual == plan
    for arm in arms:
        with (HERE / "runs" / f"{arm}_11" / "results.csv").open(encoding="utf-8-sig",newline="") as file:
            reader = csv.DictReader(file, skipinitialspace=True)
            assert [int(r["epoch"]) for r in reader] == list(range(1,16))
    return {"epochs_per_arm":15,"entries_per_arm":285,"initial_weight_sha256":plans[0]["start_sha256"],
            "same_labels_and_validation":True,"replacements":plans[1]["replacements"]}


def draw_panel(peaks):
    image = Image.open(ROOT / "experiments/blind_test_v1/frames" / FRAME).convert("RGB")
    width = 420
    thumbnail = ImageOps.contain(image, (width, 340))
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    heading = ImageFont.truetype(str(fonts/"msyhbd.ttc"), 25)
    body = ImageFont.truetype(str(fonts/"msyh.ttc"),17)
    canvas = Image.new("RGB", (3*(width+20)+20, thumbnail.height+155), "white")
    pen = ImageDraw.Draw(canvas)
    pen.text((20,8), "酸菜鱼特写  ·  三组模型在同一帧的最高分案板框", font=heading, fill="#162733")
    pen.text((20,48),"人工标注为无板。候选框最低保留分数 0.01，提示门槛 0.25。该帧此前已查看。",font=body,fill="#405065")
    for i,(model,title) in enumerate(NAMES.items()):
        x = 20+i*(width+20)
        y = 105
        pen.text((x,77), title,font=body, fill="#162733")
        canvas.paste(thumbnail,(x,y))
        peak = peaks[model]
        if peak is None:
            desc = "未输出分数达到 0.01 的框"
        else:
            left,top,right,bottom = peak["xyxy"]
            def project(v, axis):
                n = thumbnail.size[axis]
                return (x,y)[axis]+max(0,min(n-1,round(v*n/image.size[axis])))
            color = "#ef4226" if peak["score"] >= .25 else "#3f83cc"
            pen.rectangle((project(left,0),project(top,1),project(right,0),project(bottom,1)),outline=color,width=4)
            desc = f"置信度 {peak['score']:.3f}    框占画面 {peak['area']:.1%}"
        pen.text((x,y+thumbnail.height+6),desc,font=body,fill="#162733")
    canvas.save(HERE / "酸菜鱼案例.png")


def main():
    audit = audited_slots()
    evaluations = {name:load(HERE / f"eval_{name}.json") for name in NAMES}
    keys = None
    result = {}
    peaks = {}
    for name, ev in evaluations.items():
        weight = (ROOT / "training_runs/board_neg15_v1" if name == "previous_11" else HERE/"runs"/name) / "weights/last.pt"
        assert ev["weight_sha256"] == sha(weight)
        rows = ev["rows"]
        sample_keys = [(r["group"],r["name"],r["presence"]) for r in rows]
        if keys is None:
            keys = sample_keys
        assert sample_keys == keys and len(set(keys)) == len(keys) == 128
        target = next(r for r in rows if r["name"]==FRAME and r["group"]=="kitchens")
        assert target["presence"] == "no"
        peaks[name] = max(target["boxes"],key=lambda box:box["score"],default=None)
        positives = [r for r in rows if r["group"]=="kitchens" and r["presence"]=="yes"]
        usable = {str(t):sum(any(b["score"]>=t and b["area"]<.6 for b in r["boxes"]) for r in positives) for t in (.25,.5)}
        result[name] = {"weight_sha256":ev["weight_sha256"],"summary":ev["summary"],"kitchen_hits_under_area06":usable,
                        "fish_peak":peaks[name]}
    (HERE/"summary.json").write_text(json.dumps({"audit":audit,"case":FRAME,"models":result},ensure_ascii=False,indent=2),encoding="utf-8")
    draw_panel(peaks)
    lines = ["# 食物特写负例训练", "", "2026-11-16。两组均以同一份 COCO 预训练 YOLO11n 权重起训，种子 11，输入 640，15 轮，每轮 285 条。逐槽校验发现 5 处图像更换，所有标签及验证图像均一致。5 个训练视频各取 1 张，均为曾参与训练的来源。2 张保留原图，3 张裁剪，画面中可见食物与容器边沿。", "",
             "| 案板模型 | 分数门槛 | 两厨房有板检出 /4 | 其中框面积小于60% /4 | 两厨房无板误报 /15 | 开发定位 /14 | 开发误报 /70 | 白圆板 /12 | 切鸡圆板 /12 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in (.25,.5):
        for name,title in NAMES.items():
            rows = {row["group"]:row for row in result[name]["summary"] if row["threshold"]==t}
            k,d,w,c = (rows[key] for key in ["kitchens","dev","white","chicken"])
            assert (k["positives"],k["negatives"],d["positives"],d["negatives"],w["positives"],c["positives"]) == (4,15,14,70,12,12)
            useful = result[name]["kitchen_hits_under_area06"][str(t)]
            lines.append(f"| {title} | {t:g} | {k['hits']} | {useful} | {k['false_frames']} | {d['hits']} | {d['false_frames']} | {w['hits']} | {c['hits']} |")
    lines.extend(["", "两厨房 20 帧中 1 帧不确定，跳过评分。另 4 帧只有案板存在标签，小于 60% 的面积条件仅反映框的大小。开发集和圆板锚点须与参考框 IoU 达到 0.5。", "",
                  "## 酸菜鱼误报复测", "", f"该帧来自 {FRAME}，已用于设计本次实验。"])
    for name,peak in peaks.items():
        desc = f"最高分 {peak['score']:.4f}，框占画面 {peak['area']:.1%}" if peak else "未检出分数达到 0.01 的框"
        lines.append(f"- {NAMES[name]}，{desc}。")
    lines.extend(["","该例和固定开发集均为观察过的样本，结果适合检查具体退化和改善。模型只定位案板。"])
    (HERE/"报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
