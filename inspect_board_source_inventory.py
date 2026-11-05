from pathlib import Path
import json,hashlib
from ultralytics import YOLO
B=Path('.').resolve();O=B/'experiments/board_source_inventory_v1';O.mkdir(exist_ok=False)
rows=[]
for idx in [3,9]:
 p=B/f'operation_expansion_screen_v1/BV1dC7H6PEoG_{idx:02d}.jpg';rows.append({'image':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'board_present':False,'review':'full original visually checked by assistant, nonexpert','training_used':False,'group':'M1_mumu_pov_kitchen'})
(O/'reviewed_negatives.json').write_text(json.dumps(rows,indent=2));predictions=[]
for tag,w in [('default',B/'training_runs/board_v2_real/weights/best.pt'),('multisource',B/'experiments/board_multisource_train10_v1/training/weights/last.pt')]:
 m=YOLO(str(w))
 for r in rows:
  p=m.predict(r['image'],imgsz=960,rect=True,conf=.25,device=0,verbose=False)[0]
  predictions.append({'model':tag,'image':r['image'],'boxes':p.boxes.xyxy.cpu().tolist(),'conf':p.boxes.conf.cpu().tolist()})
(O/'negative_predictions.json').write_text(json.dumps(predictions,indent=2));print([(r['model'],Path(r['image']).name,len(r['boxes'])) for r in predictions])
