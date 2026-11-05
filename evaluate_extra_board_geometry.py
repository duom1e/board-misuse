import json,hashlib,os
from pathlib import Path
from PIL import Image,ImageDraw
os.environ['YOLO_OFFLINE']='True'
B=Path(__file__).resolve().parent
if __name__=='__main__':
 from ultralytics import YOLO
 O=B/'experiments/board_extra_geometry_v1';O.mkdir(exist_ok=False);records=[]
 for v,n,box in [(3,'BV1pq4y147jk_350s',[286,143,1280,720]),(4,'BV1bqsfzbELh_300s',[0,342,720,696])]:
  p=B/f'experiments/new_source_video_v{v}/frames/{n}.jpg';im=Image.open(p).convert('RGB');ImageDraw.Draw(im).rectangle(box,outline='lime',width=5);im.thumbnail((720,900));im.save(O/(n+'.jpg'))
  records.append({'image':str(p),'box':box,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'training_allowed':False,'review':'assistant approximate enclosing box; occluded edges uncertain; same sources as training, new frames only'})
 (O/'annotations.json').write_text(json.dumps(records,indent=2))
 rows=[]
 for tag,path in [('default',B/'training_runs/board_v2_real/weights/best.pt'),('multisource',B/'experiments/board_multisource_train10_v1/training/weights/last.pt')]:
  m=YOLO(str(path))
  for r in records:
   p=m.predict(r['image'],imgsz=960,rect=True,conf=.01,device=0,verbose=False)[0];g=r['box'];pred=[]
   for b,c in zip(p.boxes.xyxy.cpu().tolist(),p.boxes.conf.cpu().tolist()):
    i=max(0,min(b[2],g[2])-max(b[0],g[0]))*max(0,min(b[3],g[3])-max(b[1],g[1]));u=(b[2]-b[0])*(b[3]-b[1])+(g[2]-g[0])*(g[3]-g[1])-i;pred.append({'box':b,'conf':c,'iou':i/u})
   rows.append({'model':tag,'image':r['image'],'predictions':pred,'best_iou025':max([x['iou'] for x in pred if x['conf']>=.25],default=0)})
 (O/'results.json').write_text(json.dumps(rows,indent=2));print([(r['model'],Path(r['image']).name,round(r['best_iou025'],3)) for r in rows])
