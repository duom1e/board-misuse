"""Real detector fine-tuning with reviewed background crops and old-food replay."""
import os,json,hashlib,shutil,random
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'food_chicken_group_adaptation_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 E.mkdir(exist_ok=False)
 from ultralytics import YOLO
 rows=json.loads((B/'food_presence_patches_v3/reviewed_manifest.json').read_text(encoding='utf-8'))['records']
 neg=[r for r in rows if r['label']=='background' and r['source_group'] not in ['BV1264y1F7a6','BV1XN411q7XJ']]
 rng=random.Random(42)
 def sample(name,n):
  ps=(B/'foodseg_two_class'/name).read_text(encoding='utf-8-sig').splitlines();ps=[x for x in ps if x.strip()];rng.shuffle(ps);return ps[:n]
 old=sample('train.txt',512);val=json.loads((B/'food_balanced_detector_v1/locked_plan.json').read_text(encoding='utf-8'))['old_validation'];train=list(old)
 for r in neg:
  assert sha(r['crop'])==r['crop_sha256']
  dest=E/'images/train'/Path(r['crop']).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r['crop'],dest)
  label=E/'labels/train'/(dest.stem+'.txt');label.parent.mkdir(parents=True,exist_ok=True);label.write_text('',encoding='utf-8');train.append(str(dest))
 extra=json.loads((B/'minced_shape_annotation_v1/manifest.json').read_text(encoding='utf-8'))
 assert extra['review_status']=='assistant_overlay_reviewed_not_expert'
 positive_records=[r for r in extra['records'] if r['intended_role']=='training_candidate']
 for r in positive_records:
  assert sha(r['image'])==r['sha256'] and sha(r['label'])==r['label_sha256']
  dest=E/'images/train'/('positive_'+Path(r['image']).name);shutil.copyfile(r['image'],dest)
  label=E/'labels/train'/(dest.stem+'.txt');shutil.copyfile(r['label'],label);train.append(str(dest))
 save('positive_records.json',positive_records)
 group_manifest=json.loads((B/'food_group_boxes_v2/manifest.json').read_text(encoding='utf-8'))
 group_train=[r for r in group_manifest['records'] if r['split']=='training_candidate']
 group_held=[r for r in group_manifest['records'] if r['split']=='source_held_development']
 assert not {r['source_group'] for r in group_train}&{r['source_group'] for r in group_held}
 for r in group_train:
  assert r['review_status']=='assistant_overlay_reviewed_nonexpert'
  assert sha(r['image'])==r['sha256'] and sha(r['label'])==r['label_sha256']
  dest=E/'images/train'/('group_'+Path(r['image']).name);shutil.copyfile(r['image'],dest)
  label=E/'labels/train'/(dest.stem+'.txt');shutil.copyfile(r['label'],label);train.append(str(dest))
 save('group_training_records.json',group_train)
 save('group_held_records.json',group_held)
 assert not set(train)&set(val)
 (E/'train.txt').write_text('\n'.join(train)+'\n',encoding='utf-8');(E/'val.txt').write_text('\n'.join(val)+'\n',encoding='utf-8')
 yaml=E/'data.yaml';yaml.write_text(f'path: {E.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: meat\n  1: vegetable\n',encoding='utf-8')
 weights=B/'training_runs/foodseg_2cls/weights/best.pt';registry=sha(B/'active_models.json')
 save('locked_plan.json',dict(old_training=old,old_validation=val,group_training=group_train,group_held=group_held,positive_records=positive_records,negative_records=neg,train_count=len(train),epochs=15,optimizer='AdamW',lr0=.0001,lrf=1,seed=42,checkpoint='fixed last',backbone_trained=True,vlm_trained=False,initial_sha256=sha(weights),registry_sha256=registry,scope='Local nonexpert negative crops, not full-frame negative labels. Chicken and added pepper sources excluded from this fine-tuning; historically consumed development sources. No automatic promotion.'))
 def probe(model):
  out=[]
  for r in rows:
   pred=model.predict(r['crop'],conf=.25,imgsz=640,rect=False,device=0,verbose=False)[0]
   out.append(dict(id=r['id'],reference=r['label'],source=r['source_group'],trained_negative=r in neg,predictions=[dict(category=model.names[int(q.cls[0])],confidence=float(q.conf[0]),box=q.xyxy[0].tolist()) for q in pred.boxes]))
  return out
 model=YOLO(str(weights));save('before.json',probe(model))
 before=model.val(data=str(yaml),imgsz=640,device=0,batch=8,workers=0,plots=False,project=str(E),name='old_before',verbose=False)
 save('old_before.json',dict(map50=float(before.box.map50),map50_95=float(before.box.map)))
 model.train(data=str(yaml),epochs=15,imgsz=640,batch=8,device=0,workers=0,optimizer='AdamW',lr0=.0001,lrf=1,warmup_epochs=0,seed=42,deterministic=True,project=str(E),name='training',patience=15,plots=False,cache=False,amp=False,mosaic=0,mixup=0,fliplr=0,translate=0,scale=0,hsv_h=0,hsv_s=0,hsv_v=0,save=True)
 candidate=YOLO(str(E/'training/weights/last.pt'));save('after.json',probe(candidate))
 after=candidate.val(data=str(yaml),imgsz=640,device=0,batch=8,workers=0,plots=False,project=str(E),name='old_after',verbose=False)
 save('old_after.json',dict(map50=float(after.box.map50),map50_95=float(after.box.map)))
 assert sha(B/'active_models.json')==registry
 save('status.json',dict(stage='completed',adopted=False,default_unchanged=True))
if __name__=='__main__':main()
