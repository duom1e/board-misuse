from pathlib import Path
import json,hashlib
from ultralytics import YOLO
from train_multisource_paired15 import overlap
B=Path(__file__).resolve().parent
P=B/'experiments/multisource_paired15_20261111'
O=B/'experiments/multisource_wok15_20261112'

def main():
    O.mkdir(exist_ok=False)
    base=(P/'mixed.txt').read_text().splitlines()
    negatives=sorted((B/'experiments/wok_negative_review_20261112/images').glob('*.jpg'))
    assert len(negatives)==6
    cases=json.loads((P/'plan.json').read_text())['cases']
    held={hashlib.sha256(Path(c['image']).read_bytes()).hexdigest() for c in cases}
    assert not held.intersection(hashlib.sha256(p.read_bytes()).hexdigest() for p in negatives)
    train=base+[str(p) for p in negatives]*2
    (O/'train.txt').write_text('\n'.join(train)+'\n')
    config=O/'data.yaml'
    config.write_text(f'path: {O.as_posix()}\ntrain: train.txt\nval: {(B/"board_dataset/validation.txt").as_posix()}\nnames: [cutting_board]\n')
    (O/'plan.json').write_text(json.dumps({'epochs':15,'entries':len(train),'negative_images':[str(p) for p in negatives],'start':'default board_v2_real, fresh training','comparison':'Previous mixed15 has 236 entries; this has 248, one extra batch per epoch, so not equal-step causal comparison','cases':cases},indent=2))
    model=YOLO(str(B/'training_runs/board_v2_real/weights/best.pt'))
    model.train(data=str(config),epochs=15,batch=8,nbs=8,imgsz=640,workers=0,device=0,seed=42,optimizer='AdamW',lr0=.0001,warmup_epochs=0,amp=False,mosaic=0,plots=False,project=str(O),name='run')
    model=YOLO(str(O/'run/weights/last.pt'))
    rows=[]
    for c in cases:
        boxes=model.predict(c['image'],imgsz=960,rect=True,conf=.01,device=0,verbose=False)[0].boxes
        rows.append({'image':c['image'],'gt':c['gt'],'boxes':[{'box':b,'score':s,'iou':overlap(b,c['gt']) if c['gt'] else None} for b,s in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist())]})
    (O/'predictions.json').write_text(json.dumps(rows,indent=2))
    summary=[]
    for threshold in [.1,.25,.5]:
        summary.append({'threshold':threshold,'hits':sum(any(b['score']>=threshold and b['iou']>=.5 for b in r['boxes']) for r in rows if r['gt']),'positive_total':14,'false_frames':sum(any(b['score']>=threshold for b in r['boxes']) for r in rows if not r['gt']),'negative_total':70})
    fit=[]
    for p in negatives:
        boxes=model.predict(str(p),imgsz=960,rect=True,conf=.5,device=0,verbose=False)[0].boxes
        fit.append({'image':str(p),'false_boxes':len(boxes)})
    (O/'negative_fit.json').write_text(json.dumps(fit,indent=2))
    (O/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary,fit)
if __name__=='__main__':main()
