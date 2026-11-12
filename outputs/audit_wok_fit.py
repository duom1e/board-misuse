from pathlib import Path
import json
from PIL import Image
from ultralytics import YOLO
from train_multisource_paired15 import overlap
B=Path(__file__).resolve().parent
O=B/'experiments/multisource_wok15_20261112'

def main():
    annotations=json.loads((B/'experiments/multisource_paired15_20261111/supplement_provenance.json').read_text())
    results=[]
    for arm in ['wok']:
        model=YOLO(str(O/'run/weights/last.pt'))
        rows=[]
        for record in annotations:
            path=record['image']
            with Image.open(path) as im: w,h=im.size
            _,x,y,bw,bh=map(float,record['label_text'].split())
            gt=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
            prediction=model.predict(path,imgsz=640,rect=True,conf=.01,device=0,verbose=False)[0].boxes
            boxes=[{'box':b,'confidence':c,'iou':overlap(b,gt)} for b,c in zip(prediction.xyxy.cpu().tolist(),prediction.conf.cpu().tolist())]
            rows.append({'image':path,'source':Path(path).stem.split('_')[0],'gt':gt,'boxes':boxes})
        for threshold in [.1,.25,.5]:
            groups={}
            for row in rows:
                group=groups.setdefault(row['source'],{'hits':0,'total':0})
                group['total']+=1
                group['hits']+=int(any(b['confidence']>=threshold and b['iou']>=.5 for b in row['boxes']))
            results.append({'arm':arm,'threshold':threshold,'groups':groups,'hits':sum(g['hits'] for g in groups.values()),'total':len(rows)})
        (O/f'{arm}_training_fit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (O/'training_fit_summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results))
if __name__=='__main__': main()
