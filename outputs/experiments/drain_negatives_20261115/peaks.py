"""Render numbered contact sheets for all candidate-event peaks at a chosen window."""
import argparse
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw
from train import HERE, VIDEO_ROOT, read


def render(arm, source, window):
    result = read(HERE / "evaluation" / f"{arm}.json")["videos"][source]
    events = result["windows"][str(window)]["episodes"]
    folder = HERE / "evaluation" / "peaks"
    folder.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(next(VIDEO_ROOT.glob(f"*{source}*.mp4"))))
    try:
        for page in range(0,len(events),12):
            entries = events[page:page+12]
            sheet = Image.new("RGB",(900, math.ceil(len(entries)/3)*212),"white")
            for i, entry in enumerate(entries):
                frame = entry["peak_frame"]
                assert cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
                ok, bgr = cap.read()
                assert ok
                im = Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(im)
                draw.rectangle(entry["peak_box"], outline="#ec4637", width=max(3,im.width//280))
                im.thumbnail((296,178))
                x, y = i%3*300, i//3*212
                sheet.paste(im,(x,y))
                ImageDraw.Draw(sheet).text((x,y+180),f"#{page+i+1}  t={frame/result['meta']['fps']:.2f}s  conf={entry['peak_confidence']:.3f}",fill="black")
            dst = folder / f"{arm}_{source}_w{window}_{page//12+1}.jpg"
            sheet.save(dst,quality=90)
            print(dst)
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arms",nargs="+")
    parser.add_argument("--window",type=int,default=5,choices=(1,5))
    args = parser.parse_args()
    for arm in args.arms:
        for source in ("BV1bY4y1Y7v5","BV1b1ixB5ERA"):
            render(arm,source,args.window)
