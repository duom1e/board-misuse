from pathlib import Path
import json
from ultralytics import YOLO
from train_multisource_paired15 import overlap
B=Path(__file__).resolve().parent
O=B/'experiments/wok15_scale_probe_20261112'
def main():
    O.mkdir(exist_ok=False)
    cases=json.loads((B/'experiments/multisource_paired15_20261111/plan.json').read_text())['cases']
    model=YOLO(str(B/'experiments/multisource_wok15_20261112/run/weights/last.pt'))
    summary=[]
    for size in [640,1280]:
        rows=[]
        for c in cases:
            pred=model.predict(c['image'],imgsz=size,rect=True,conf=.01,device=0,verbose=False)[0].boxes
            rows.append({'image':c['image'],'gt':c['gt'],'boxes':[{'box':b,'score':s,'iou':overlap(b,c['gt']) if c['gt'] else None} for b,s in zip(pred.xyxy.cpu().tolist(),pred.conf.cpu().tolist())]})
        (O/f'{size}.json').write_text(json.dumps(rows,indent=2))
        for threshold in [.1,.25,.5]:
            summary.append({'size':size,'threshold':threshold,'hits':sum(any(b['score']>=threshold and b['iou']>=.5 for b in r['boxes']) for r in rows if r['gt']),'false_frames':sum(any(b['score']>=threshold for b in r['boxes']) for r in rows if not r['gt'])})
    (O/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary)
if __name__=='__main__':main()
