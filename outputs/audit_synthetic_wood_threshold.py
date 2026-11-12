from pathlib import Path
import json
from ultralytics import YOLO
B=Path(__file__).resolve().parent
P=B/'experiments/synthetic_board_pilot_v1'
V=B/'experiments/synthetic_board_occlusion_v2'
O=B/'experiments/synthetic_wood_threshold_v1'
def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return overlap/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)
def main():
    O.mkdir(exist_ok=False)
    cases=json.loads((P/'real_predictions.json').read_text())
    for row in json.loads((V/'display_board_check.json').read_text()):
        if row['model']=='real':
            cases.append(dict(image=str(B/f"animal_operation_screen_v1/BV1TS4y1K7ZH_{row['frame']}.jpg"),gt=row['box']))
    assert len(cases)==75 and sum(c['gt'] is not None for c in cases)==5
    for record in json.loads((B/'experiments/board_manual_v2/manifest.json').read_text())['records']:
        if record['split']=='val': cases.append(dict(image=record['image'],gt=record['box_px']))
    summary=[]
    for tag,run in [('real',P/'real_run'),('synthetic',P/'mixed_run'),('occlusion',V/'mixed_run'),('wood',B/'experiments/synthetic_board_wood_v3/mixed_run')]:
        model=YOLO(str(run/'weights/last.pt'));rows=[]
        for case in cases:
            pred=model.predict(case['image'],imgsz=960,rect=True,conf=.01,device=0,verbose=False)[0].boxes
            boxes=[dict(box=b,score=c,iou=iou(b,case['gt']) if case['gt'] else None) for b,c in zip(pred.xyxy.cpu().tolist(),pred.conf.cpu().tolist())]
            rows.append(dict(image=case['image'],gt=case['gt'],boxes=boxes))
        for t in [.01,.1,.25,.5]:
            hits=sum(any(b['score']>=t and b['iou']>=.5 for b in r['boxes']) for r in rows if r['gt'])
            false=sum(any(b['score']>=t for b in r['boxes']) for r in rows if not r['gt'])
            summary.append(dict(model=tag,threshold=t,hits=hits,false_frames=false))
        (O/f'{tag}.json').write_text(json.dumps(rows,indent=2))
    (O/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
