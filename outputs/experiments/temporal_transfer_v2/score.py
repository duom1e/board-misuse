"""Score locked confidence and same-location persistence on two transfer videos."""
import hashlib
import json
from collections import deque
from pathlib import Path

OUT = Path(__file__).resolve().parent
PROJECT = Path("D:/CodexProjects/board-misuse/outputs")
CHICKEN = "BV1264y1F7a6"
CANTEEN = "BV1bY4y1Y7v5"
CONFIDENCE = 0.25
IOU = 0.5
WINDOWS = (1, 5)
REFINED = {0: [52, 0, 1216, 720], 7: [59, 0, 1280, 720]}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    common = width * height
    total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - common
    return common / total if total else 0.0


def load_anchors(meta):
    sources = read_json(PROJECT / "chicken_presence_sources_v1/sources.json")
    original = next(item for item in sources if item["id"] == CHICKEN)
    assert meta["video_sha256"] == original["sha256"]
    labels = read_json(PROJECT / "experiments/chicken_block_full12_20261115/annotations.json")
    by_name = {Path(row["image"]).name: row["box"] for row in labels}
    anchors = {}
    for row in original["frames"]:
        image = Path(row["image"])
        assert row["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
        anchors[row["frame_index"]] = {
            "label_index": row["index"],
            "name": image.name,
            "reference": REFINED.get(row["index"], by_name[image.name]),
            "revised_on_original": row["index"] in REFINED,
        }
    assert len(anchors) == 12
    return anchors


def score(frames, meta, anchors, window):
    recent = deque(maxlen=window)
    marked = 0
    scored_anchors = []
    episodes = []
    current = None
    fps = meta["fps"]
    for offset, predictions in enumerate(frames):
        frame = meta["start_frame"] + offset
        boxes = [p for p in predictions if p["confidence"] >= CONFIDENCE]
        recent.append(boxes)
        kept = [
            p for p in boxes
            if len(recent) == window and all(
                any(iou(p["xyxy"], previous["xyxy"]) >= IOU for previous in group)
                for group in recent
            )
        ]
        if frame in anchors:
            label = anchors[frame]
            best = max((iou(p["xyxy"], label["reference"]) for p in kept), default=0)
            scored_anchors.append({**label, "frame": frame, "best_iou": best, "hit": best >= IOU})
        if not kept:
            current = None
            continue
        marked += 1
        peak = max(kept, key=lambda p: p["confidence"])
        if current is None:
            current = {"start_frame": frame, "peak_confidence": -1}
            episodes.append(current)
        current["end_frame_inclusive"] = frame
        if peak["confidence"] > current["peak_confidence"]:
            current.update(peak_frame=frame, peak_box=peak["xyxy"], peak_confidence=peak["confidence"])
    assert len(scored_anchors) == len(anchors)
    for episode in episodes:
        episode["start_sec"] = episode["start_frame"] / fps
        episode["end_sec_exclusive"] = (episode["end_frame_inclusive"] + 1) / fps
        episode["duration_sec"] = (episode["end_frame_inclusive"] + 1 - episode["start_frame"]) / fps
    return {
        "window": window, "flagged_frames": marked, "episode_count": len(episodes),
        "episodes": episodes, "anchors": scored_anchors,
        "anchor_hits": sum(a["hit"] for a in scored_anchors),
    }


def main():
    data = {"confidence": CONFIDENCE, "iou": IOU, "sources": {}}
    for source in (CHICKEN, CANTEEN):
        folder = OUT / source
        meta = read_json(folder / "meta.json")
        frames = read_json(folder / "positive_frames.json")
        assert len(frames) == meta["frames"] and meta["confidence"] == CONFIDENCE
        anchors = load_anchors(meta) if source == CHICKEN else {}
        data["sources"][source] = {
            "meta": meta, "comparisons": [score(frames, meta, anchors, window) for window in WINDOWS]
        }
    chicken, canteen = (data["sources"][s] for s in (CHICKEN, CANTEEN))
    assert chicken["meta"]["weight_sha256"] == canteen["meta"]["weight_sha256"]
    lines = [
        "# 连续帧规则的跨视频复测", "",
        "沿用上一轮补图模型与固定参数。置信度至少 0.25，当前框须与最近每帧至少一个候选框的 IoU 达到 0.5，含当前帧。按整幅画面连续有框合并事件。", "",
        "| 连续帧数 | 食堂片段候选段数 | 食堂片段有框帧数 | 切鸡圆板命中 / 12 |",
        "|---:|---:|---:|---:|",
    ]
    for pos, neg in zip(chicken["comparisons"], canteen["comparisons"]):
        lines.append(f"| {pos['window']} | {neg['episode_count']} | {neg['flagged_frames']} | {pos['anchor_hits']} / 12 |")
    lines.extend(["", "切鸡原视频的观测帧通过原有清单的帧号及图像哈希对应。00、07 使用原图复核框，其他十张使用先前粗框。帧按原视频从零起算，事件结束秒数为不含该时刻。", "",
                  "食堂视频只复测选定片段。表中统计模型给出的候选时段；判定误报需结合实际画面。两段视频的结果仅说明案板定位表现。"])
    (OUT / "score.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (OUT / "计分报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for pos, neg in zip(chicken["comparisons"], canteen["comparisons"]):
        print(pos["window"], pos["anchor_hits"], neg["episode_count"], neg["flagged_frames"])


if __name__ == "__main__":
    main()
