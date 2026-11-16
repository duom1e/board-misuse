"""Recreate visually reviewed no-board food close-ups from exact local video frames."""
import hashlib
import json
import os
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BOARD_OUTPUTS", "D:/CodexProjects/board-misuse/outputs"))


def prepare():
    samples = json.loads((HERE / "samples.json").read_text(encoding="utf-8"))
    (HERE / "images").mkdir(exist_ok=True)
    (HERE / "labels").mkdir(exist_ok=True)
    rows = []
    for record in samples:
        source = record["source"]
        frame_index = record["frame"]
        videos = list((ROOT.parent / "待导入视频").glob(f"*{source}*.mp4"))
        if len(videos) != 1:
            raise ValueError(f"Expected one original video for {source}")
        with videos[0].open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != record["video_sha256"]:
                raise ValueError(f"Original video checksum mismatch for {source}")
        cap = cv2.VideoCapture(str(videos[0]))
        try:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise ValueError(f"Could not seek {source} to {frame_index}")
            ok, frame = cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index + 1:
                raise ValueError(f"Frame {frame_index} unavailable in {source}")
        finally:
            cap.release()
        if record["crop"]:
            left, top, right, bottom = record["crop"]
            if not (0 <= left < right <= frame.shape[1] and 0 <= top < bottom <= frame.shape[0]):
                raise ValueError(f"Crop outside original image for {source}")
            frame = frame[top:bottom, left:right]
        name = f"{source}_f{frame_index}"
        ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise OSError(f"JPEG encoding failed for {name}")
        data = image.tobytes()
        (HERE / "images" / f"{name}.jpg").write_bytes(data)
        (HERE / "labels" / f"{name}.txt").write_bytes(b"")
        rows.append({"image": f"{name}.jpg", "width": frame.shape[1], "height": frame.shape[0],
                     "sha256": hashlib.sha256(data).hexdigest()})
    (HERE / "prepared.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Prepared {len(rows)} no-board images", flush=True)


if __name__ == "__main__":
    prepare()
