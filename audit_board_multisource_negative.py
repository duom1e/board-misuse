import json,os
from pathlib import Path
os.environ['YOLO_OFFLINE']='True'
from ultralytics import YOLO
B=Path(__file__).resolve().parent
if __name__=='__main__':
 O=B/'experiments/board_multisource_train10_v1';rows=[]
 for tag,weight in [('default',B/'training_runs/board_v2_real/weights/best.pt'),('multisource10',O/'training/weights/last.pt')]:
  model=YOLO(str(weight))
  for v in range(1,6):
   folder=B/f'experiments/new_source_video_v{v}'
   labels=json.loads((folder/'labels_blind.json').read_text(encoding='utf-8'))['labels']
   for name,label in labels.items():
    if label.get('usable') and label.get('board_present')=='no':
     p=model.predict(str(folder/'frames'/name),imgsz=960,rect=True,conf=.5,device=0,verbose=False)[0]
     rows.append({'model':tag,'image':str(folder/'frames'/name),'boxes':p.boxes.xyxy.cpu().tolist(),'conf':p.boxes.conf.cpu().tolist()})
 summary={tag:{'frames':sum(r['model']==tag for r in rows),'false_frames':sum(r['model']==tag and bool(r['boxes']) for r in rows)} for tag in ['default','multisource10']}
 (O/'negative_audit.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2),encoding='utf-8');print(summary)
