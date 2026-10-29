import os,json,hashlib
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'food_batch36_v2'
def main():
 from ultralytics import YOLO
 cases=json.loads((E/'locked_plan.json').read_text(encoding='utf-8'))['cases'];cfg=json.loads((B/'active_models.json').read_text(encoding='utf-8'))['foodseg'];model=YOLO(cfg['weights']);rows=[]
 for i,c in enumerate(cases):
  p=model.predict(c['image'],imgsz=640,rect=False,conf=.25,device='cpu',verbose=False)[0]
  rows.append(dict(index=i,source=c['source'],seconds=c['seconds'],predictions=[dict(box=q.xyxy[0].tolist(),category=model.names[int(q.cls[0])],confidence=float(q.conf[0])) for q in p.boxes]))
 (E/'yolo_food.json').write_text(json.dumps(dict(rows=rows,weights=cfg['weights'],weights_sha256=hashlib.sha256(Path(cfg['weights']).read_bytes()).hexdigest(),conf=.25,rect=False,imgsz=640,device='cpu',training_performed=False),indent=2),encoding='utf-8');print('Completed YOLO food comparison:',len(rows))
if __name__=='__main__':main()
