"""Fixed mixed positive/background development cohort; no prediction overrides."""
import os,json,hashlib
from pathlib import Path
from PIL import Image
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'detector_vlm_category_chain_v1'
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 E.mkdir(exist_ok=False)
 groups=json.loads((B/'food_group_boxes_v1/manifest.json').read_text(encoding='utf-8'))['records'];patch=json.loads((B/'food_presence_patches_v3/reviewed_manifest.json').read_text(encoding='utf-8'))['records'];cases=[dict(id=r['id'],image=r['image'],kind='food_group',category=r['category'],boxes=r['boxes_xyxy'],source=r['source_group']) for r in groups]
 cases += [dict(id=r['id'],image=r['crop'],kind='background',category='background',boxes=[],source=r['source_group']) for r in patch if r['label']=='background']
 for c in cases:c['sha256']=hashlib.sha256(Path(c['image']).read_bytes()).hexdigest()
 weights={'original':B/'training_runs/foodseg_2cls/weights/best.pt','yellow_candidate':B/'food_yellow_chicken_adaptation_v1/training/weights/last.pt'}
 save('locked_plan.json',dict(cases=cases,weights={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in weights.items()},conf=.25,imgsz=640,rect=False,scope='29 consumed development images; 6 group ROIs +23 background patches. Candidate trained on some images/source crops. Not independent evaluation or deployment.'))
 from ultralytics import YOLO
 proposals=[]
 for name,w in weights.items():
  model=YOLO(str(w))
  for c in cases:
   result=model.predict(c['image'],imgsz=640,rect=False,conf=.25,device='cpu',verbose=False)[0]
   proposals.append(dict(model=name,case=c,predictions=[dict(box=q.xyxy[0].tolist(),original_category=model.names[int(q.cls[0])],confidence=float(q.conf[0])) for q in result.boxes]))
 save('detector_results.json',proposals)
 from region_vlm_runtime import LocalRegionModel
 from food_category_review import FOOD_CATEGORY_PROMPT,review_board_foods
 model=LocalRegionModel();out=[]
 for r in proposals:
  im=Image.open(r['case']['image']).convert('RGB')
  for q in r['predictions']:
   opinion=review_board_foods(im,[dict(box=q['box'],location='board_candidate')],lambda crop:model.predict(crop,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12))[0];opinion.pop('location');q['review']=opinion;q['original_category_preserved']=True
  out.append(r);save('partial_results.json',out);print(r['model'],r['case']['id'],len(r['predictions']),flush=True)
 save('results.json',dict(rows=out,training_performed=False,adopted=False))
if __name__=='__main__':main()
