"""Compare both candidate models across640/960 without changing deployment."""
import os,json
from pathlib import Path
os.environ['YOLO_OFFLINE']='True'
from ultralytics import YOLO
from evaluate_food_group_boxes import iou
B=Path(__file__).resolve().parent;E=B/'fullmask_fit_comparison_v1'
def main():
 E.mkdir(exist_ok=False);out=[]
 for folder in ['food_clean_multiscale45_v1','food_fullmask_multiscale45_v1']:
  m=YOLO(str(B/folder/'training/weights/last.pt'))
  for size in [640,960]:
   for ds in ['meat_operation_replay_v1','pov_operation_replay_v1','chopping_chicken_replay_v1']:
    records=json.loads((B/ds/'manifest.json').read_text())['records'];tp=total=0
    for r in records:
     p=m.predict(r['image'],conf=.25,imgsz=size,rect=False,device='cpu',verbose=False)[0];ps=[(q.xyxy[0].tolist(),m.names[int(q.cls[0])]) for q in p.boxes];used=set();cat='vegetable' if ds.startswith('pov') else 'meat'
     for g in r['boxes']:
      total+=1;opts=[(iou(g,b),i) for i,(b,c) in enumerate(ps) if i not in used and c==cat and iou(g,b)>=.5]
      if opts:used.add(max(opts)[1]);tp+=1
    out.append(dict(model=folder,size=size,dataset=ds,tp=tp,total=total))
 (E/'results.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
