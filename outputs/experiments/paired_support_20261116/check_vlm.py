"""Run the project's frozen Qwen board prompt on exactly the same decoded kitchen pixels."""
import json
import sys
import argparse

import cv2
from PIL import Image

from evaluate import inputs, iou
from train import HERE, ROOT, sha

sys.path.insert(0,str(ROOT))
from review_combined import BOARD_PROMPT, ask_vlm, load_vlm, parse_vlm_boards, validate_region_response


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--regression",action="store_true")
    args=parser.parse_args()
    groups=("dev","white","chicken") if args.regression else ("kitchens","transfer","dishes")
    data=[(r,image) for r,image in inputs() if r["group"] in groups]
    assert len(data)==(108 if args.regression else 30)
    vlm=load_vlm()
    rows=[]
    for r,pixels in data:
        im=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB))
        raw=ask_vlm(im,vlm)
        normalized,error=parse_vlm_boards(raw)
        valid=validate_region_response(json.dumps({"boards":normalized,"containers":[],"food":[]}))
        boxes=[]
        if valid["valid"]:
            w,h=im.size
            boxes=[[a*w/1000,b*h/1000,c*w/1000,d*h/1000] for a,b,c,d in normalized]
        rows.append({**r,"raw":raw,"valid":valid["valid"],"error":error or valid.get("errors_by_code"),
            "boxes":boxes,"best_iou":max((iou(b,r["gt"]) for b in boxes),default=0) if r["gt"] else None})
        print(r["group"],r["name"],len(boxes),valid["valid"],flush=True)
    summary={}
    for group in groups:
        subset=[r for r in rows if r["group"]==group]
        summary[group]={
            "total":len(subset),"valid":sum(r["valid"] for r in subset),
            "presence_hits":sum(bool(r["boxes"]) for r in subset if r["presence"]=="yes"),
            "localized":sum(r["best_iou"]>=.5 for r in subset if r["gt"]),
            "false_frames":sum(bool(r["boxes"]) for r in subset if r["presence"]=="no")}
    dest="vlm_regression.json" if args.regression else "vlm.json"
    (HERE/dest).write_text(json.dumps({"revision":vlm["revision"],"prompt_sha256":sha(BOARD_PROMPT),
        "summary":summary,"rows":rows},indent=2,ensure_ascii=False),encoding="utf-8")


if __name__=="__main__":
    main()
