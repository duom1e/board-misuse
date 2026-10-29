"""Compare fixed detector and VLM on visually reviewed patch references."""
import argparse,json,hashlib,os
from pathlib import Path
from PIL import Image
B=Path(__file__).resolve().parent;E=B/'food_presence_patches_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['yolo','vlm']);a=p.parse_args()
 manifest=E/'reviewed_manifest.json';data=json.loads(manifest.read_text(encoding='utf-8'))
 out=E/f'{a.mode}_results.json';partial=E/f'{a.mode}_partial.json'
 assert not out.exists() and not partial.exists()
 if a.mode=='yolo':
  os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
  from ultralytics import YOLO
  weights=json.loads((B/'active_models.json').read_text(encoding='utf-8'))['foodseg']['weights'];model=YOLO(weights)
  config=dict(weights_sha256=sha(weights),imgsz=640,rect=False,conf=.25,device='cpu')
 else:
  from region_vlm_runtime import LocalRegionModel
  from food_category_review import FOOD_CATEGORY_PROMPT,CATEGORIES
  model=LocalRegionModel();config=dict(revision=model.revision,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12,thumbnail=512,crop_margin=0)
 rows=[]
 for r in data['records']:
  assert sha(r['crop'])==r['crop_sha256'] and sha(r['image'])==r['image_sha256']
  row=dict(id=r['id'],reference=r['label'],source_group=r['source_group'])
  if a.mode=='yolo':
   pred=model.predict(r['crop'],imgsz=640,rect=False,conf=.25,device='cpu',verbose=False)[0]
   row['predictions']=[dict(category=model.names[int(q.cls[0])],confidence=float(q.conf[0]),box=q.xyxy[0].tolist()) for q in pred.boxes]
  else:
   raw=model.predict(Image.open(r['crop']),prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12)
   row.update(raw_response=raw,opinion=raw.strip().lower() if raw.strip().lower() in CATEGORIES else 'unknown',response_valid=raw.strip().lower() in CATEGORIES)
  rows.append(row);save(partial,dict(rows=rows));print(row['id'],row.get('opinion',row.get('predictions')),flush=True)
 save(out,dict(rows=rows,configuration=config,manifest_sha256=sha(manifest),training_performed=False))
if __name__=='__main__':main()
