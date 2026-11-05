import json,os
from pathlib import Path
os.environ['YOLO_OFFLINE']='True'
from ultralytics import YOLO
B=Path(__file__).resolve().parent
if __name__=='__main__':
 O=B/'experiments/board_positive_source_audit_v2';O.mkdir(exist_ok=False);rows=[]
 models={'default':B/'training_runs/board_v2_real/weights/best.pt','positive10':B/'experiments/board_manual_multiscale10_v1/training/weights/last.pt','negative10':B/'experiments/board_negative_multiscale10_v1/training/weights/last.pt'}
 for tag,path in models.items():
  model=YOLO(str(path))
  for v in range(1,6):
   folder=B/f'experiments/new_source_video_v{v}'
   for name,label in json.loads((folder/'labels_blind.json').read_text(encoding='utf-8'))['labels'].items():
    if label.get('usable') and label.get('board_present')=='yes':
     pred=model.predict(str(folder/'frames'/name),imgsz=960,rect=True,conf=.25,device=0,verbose=False)[0]
     rows.append({'model':tag,'source':name.split('_')[0],'image':str(folder/'frames'/name),'boxes':pred.boxes.xyxy.cpu().tolist(),'conf':pred.boxes.conf.cpu().tolist()})
 summary=[]
 for tag in models:
  for source in sorted({r['source'] for r in rows}):
   g=[r for r in rows if r['model']==tag and r['source']==source]
   summary.append({'model':tag,'source':source,'frames':len(g),'any_candidate_025':sum(bool(r['boxes']) for r in g),'any_candidate_050':sum(any(c>=.5 for c in r['conf']) for r in g),'training_source':source=='BV12a41187Ja'})
 (O/'results.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2),encoding='utf-8');print(json.dumps(summary))

