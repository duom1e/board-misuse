import json, os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
B=Path(__file__).resolve().parent
O=B/'experiments/board_counter_round_v1'
def main():
    base=(B/'board_dataset/train.txt').read_text().splitlines()
    synthetic=json.loads((O/'supplement.json').read_text())*4
    cases=json.loads((B/'experiments/board_tony_geometry_v1/annotations.json').read_text())
    negative_paths=[r['image'] for r in json.loads((B/'experiments/board_balanced_train10_v1/negative_audit.json').read_text())['rows'] if r['model']=='default']
    assert len(negative_paths)==70
    totals={}
    for arm,extra in [('mixed',synthetic)]:
        (O/f'{arm}.txt').write_text('\n'.join(base+extra)+'\n')
        yaml=O/f'{arm}.yaml'
        yaml.write_text(f'path: {O.as_posix()}\ntrain: {arm}.txt\nval: {(B/"board_dataset/validation.txt").as_posix()}\nnames: [cutting_board]\n')
        model=YOLO(str(B/'training_runs/board_v2_real/weights/best.pt'))
        model.train(data=str(yaml),epochs=5,batch=8,nbs=8,imgsz=640,workers=0,device=0,seed=42,optimizer='AdamW',lr0=.0001,warmup_epochs=0,amp=False,mosaic=0,plots=False,project=str(O),name=arm+'_run')
        detector=YOLO(str(O/(arm+'_run')/'weights/last.pt'))
        hits=fp=0;rows=[]
        for image,gt in [(r['image'],r['box']) for r in cases]+[(p,None) for p in negative_paths]:
            boxes=detector.predict(image,imgsz=960,rect=True,conf=.5,device=0,verbose=False)[0].boxes.xyxy.cpu().tolist()
            rows.append(dict(image=image,boxes=boxes,gt=gt))
            if gt is None: fp+=bool(boxes);continue
            for a in boxes:
                inter=max(0,min(a[2],gt[2])-max(a[0],gt[0]))*max(0,min(a[3],gt[3])-max(a[1],gt[1]))
                if inter/((a[2]-a[0])*(a[3]-a[1])+(gt[2]-gt[0])*(gt[3]-gt[1])-inter)>=.5:
                    hits+=1;break
        totals[arm]=dict(round_board_hits=hits,negative_false_frames=fp)
        (O/f'{arm}_predictions.json').write_text(json.dumps(rows))
        (O/'summary.json').write_text(json.dumps(totals,indent=2))
if __name__=='__main__':main()
