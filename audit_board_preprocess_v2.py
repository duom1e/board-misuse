import json,os
from pathlib import Path
os.environ['YOLO_OFFLINE']='True'
from ultralytics import YOLO
B=Path(__file__).resolve().parent

def main():
 O=B/'experiments/board_preprocess_audit_v2';O.mkdir(exist_ok=False)
 records=json.loads((B/'experiments/board_manual_v2/manifest.json').read_text())['records'];rows=[]
 for tag,path in [('default',B/'training_runs/board_v2_real/weights/best.pt'),('manual10',B/'experiments/board_manual_train10_v2/training/weights/last.pt'),('multiscale10',B/'experiments/board_manual_multiscale10_v1/training/weights/last.pt')]:
  model=YOLO(str(path))
  for size in [640,960]:
   for rect in [True,False]:
    for r in records:
     p=model.predict(r['image'],imgsz=size,rect=rect,conf=.01,device=0,verbose=False)[0];gt=r['box_px'];boxes=[]
     for b,c in zip(p.boxes.xyxy.cpu().tolist(),p.boxes.conf.cpu().tolist()):
      inter=max(0,min(b[2],gt[2])-max(b[0],gt[0]))*max(0,min(b[3],gt[3])-max(b[1],gt[1]));iou=inter/((b[2]-b[0])*(b[3]-b[1])+(gt[2]-gt[0])*(gt[3]-gt[1])-inter)
      boxes.append({'box':b,'conf':c,'iou':iou})
     rows.append({'model':tag,'size':size,'rect':rect,'image':r['image'],'split':r['split'],'predictions':boxes})
 summary=[]
 for tag in ['default','manual10','multiscale10']:
  for size in [640,960]:
   for rect in [True,False]:
    for conf in [.25,.5]:
     group=[r for r in rows if (r['model'],r['size'],r['rect'])==(tag,size,rect) and r['split']=='val']
     summary.append({'model':tag,'size':size,'rect':rect,'conf':conf,'hits':sum(any(p['iou']>=.5 and p['conf']>=conf for p in r['predictions']) for r in group),'total':len(group)})
 (O/'results.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2),encoding='utf-8')
 print(json.dumps(summary))
if __name__=='__main__':main()
