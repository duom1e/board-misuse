import json, os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
B=Path(__file__).resolve().parent
O=B/'experiments/synthetic_board_occlusion_v2'
def main():
    O.mkdir(exist_ok=False)
    rng=np.random.default_rng(42)
    base=(B/'board_dataset/train.txt').read_text().splitlines()
    negatives=json.loads((B/'experiments/board_source_inventory_v1/reviewed_negatives.json').read_text())
    synthetic=[]
    for folder in ['images','labels','masks']:
        (O/folder).mkdir()
    for i in range(16):
        img=cv2.resize(cv2.imread(negatives[i%2]['image']),(640,360))
        x,y=int(rng.integers(180,460)),int(rng.integers(130,250))
        mask=np.zeros((360,640),np.uint8)
        cv2.ellipse(mask,(x,y),(int(rng.integers(80,155)),int(rng.integers(52,105))),float(rng.uniform(-20,20)),0,360,255,-1)
        yy,xx=np.indices(mask.shape)
        grain=6*np.sin(np.sqrt(((xx-x)*.75)**2+(yy-y)**2)*.55)+rng.normal(0,5,mask.shape)
        wood=np.clip(np.array([33,52,76])+grain[:,:,None]+rng.integers(-10,20),0,255).astype(np.uint8)
        # Approximate foreground silhouettes traced on the two 640x360 backgrounds.
        polygons = [
            [[(200,355),(198,311),(227,254),(240,210),(251,203),(272,207),(281,231),(281,247),(261,264),(250,295),(231,360)],
             [(640,170),(543,128),(491,99),(460,75),(443,60),(420,57),(410,72),(419,91),(448,110),(478,136),(549,183),(640,221)]],
            [[(159,360),(165,320),(166,285),(171,255),(180,247),(199,244),(213,252),(216,270),(205,306),(200,360)],
             [(640,339),(526,282),(452,239),(387,204),(365,195),(349,196),(343,205),(356,217),(379,219),(431,252),(512,302),(604,360),(640,360)]]
        ]
        foreground = np.zeros_like(mask)
        for polygon in polygons[i % 2]:
            cv2.fillPoly(foreground,[np.array(polygon,np.int32)],255)
        visible = cv2.bitwise_and(mask,cv2.bitwise_not(foreground))
        img[visible>0]=wood[visible>0]
        cv2.imwrite(str(O/'masks'/f'{i}_visible.png'),visible)
        left,top,w,h=cv2.boundingRect(mask)
        path=O/'images'/f'{i}.jpg'
        cv2.imwrite(str(path),img);cv2.imwrite(str(O/'masks'/f'{i}.png'),mask)
        (O/'labels'/f'{i}.txt').write_text(f'0 {(left+w/2)/640} {(top+h/2)/360} {w/640} {h/360}\n')
        synthetic.append(str(path))
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
