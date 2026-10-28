"""Training-fit diagnostic ONLY: train=val, no generalization assessment."""
import json,hashlib,shutil
from pathlib import Path
from ultralytics import YOLO
from run_canteen_food_detector import evaluate
from run_food_mixed_selection import operational
B=Path(__file__).resolve().parent;E=B/'experiments/minced_training_fit_check'
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
 E.mkdir(exist_ok=False);d=json.loads((B/'minced_shape_annotation_v1/manifest.json').read_text(encoding='utf-8'));assert d['review_status']=='assistant_overlay_reviewed_not_expert'
 rows=[r for r in d['records'] if r['intended_role']=='training_candidate'];assert len(rows)==11
 for r in rows:
  ip=E/'images/train'/Path(r['image']).name;lp=E/'labels/train'/(ip.stem+'.txt');ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(r['image'],ip);shutil.copyfile(r['label'],lp);assert digest(ip)==r['sha256'] and digest(lp)==r['label_sha256']
 weights=B/'training_runs/foodseg_2cls/weights/best.pt';registry=digest(B/'active_models.json')
 plan={'scope':'Training memorization/fit diagnostic only. SAME11 train and val, no validation or independent accuracy claim. Reserved source/B20/old regression NOT evaluated.','images':[r['image'] for r in rows],'source_manifest_sha256':digest(B/'minced_shape_annotation_v1/manifest.json'),'initial_weights':str(weights),'initial_sha256':digest(weights),'epochs':120,'batch':11,'nbs':11,'optimizer':'AdamW','lr0':.001,'lrf':1.,'warmup_epochs':0,'freeze':0,'augmentations':'mosaic/mixup/scale/translate/perspective/shear/degrees/flip/hsv all0; no augmentation','imgsz':640,'seed':42,'selection':'Fixed last.pt after120 epochs; no checkpoint search, no threshold search.','conf':.5,'match_iou':.5,'fit_gates':{'precision':.9,'recall':.9,'each_class_recall':.8},'registry_before_sha256':registry,'promotion':False}
 write(E/'locked_plan.json',plan)
 yaml=E/'data.yaml';yaml.write_text(f'path: {E.as_posix()}\ntrain: images/train\nval: images/train\nnames:\n  0: meat\n  1: vegetable\n',encoding='utf-8')
 model=YOLO(str(weights));before=evaluate(model,rows);write(E/'before.json',before)
 model.train(data=str(yaml),epochs=120,imgsz=640,batch=11,nbs=11,device=0,workers=0,optimizer='AdamW',lr0=.001,lrf=1.,warmup_epochs=0,freeze=0,seed=42,deterministic=True,weight_decay=.0005,project=str(E),name='training',val=False,patience=120,plots=False,cache=False,amp=True,mosaic=0,mixup=0,close_mosaic=0,fliplr=0,flipud=0,translate=0,scale=0,perspective=0,shear=0,degrees=0,hsv_h=0,hsv_s=0,hsv_v=0,save=True)
 after=evaluate(YOLO(str(E/'training/weights/last.pt')),rows);score=operational(after)
 passed=(after['counts']['precision'] or 0)>=.9 and after['counts']['recall']>=.9 and all(r['recall']>=.8 for r in score['classes'].values())
 write(E/'result.json',{'before':before,'after':after,'after_per_class':score['classes'],'training_fit_gate_passed':passed,'scope':plan['scope'],'adopted':False,'registry_unchanged':digest(B/'active_models.json')==registry})
 print(json.dumps({'before':before['counts'],'after':after['counts'],'class_recall':score['classes'],'fit_gate':passed}))
