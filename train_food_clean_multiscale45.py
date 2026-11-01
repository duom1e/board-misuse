"""Fresh generic80-class initialization, clean two-class training, fixed45 epochs."""
import os,json,hashlib,shutil
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
class AuditedTrainer(DetectionTrainer):
 def preprocess_batch(self,batch):
  batch=super().preprocess_batch(batch)
  key=str(tuple(batch["img"].shape[-2:]));self.size_counts=getattr(self,"size_counts",{});self.size_counts[key]=self.size_counts.get(key,0)+1
  return batch
B=Path(__file__).resolve().parent;E=B/'food_clean_multiscale45_v1'
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);parent=B/'food_empty_label_clean_training_v1'
 for n in ['train.txt','val.txt']:shutil.copyfile(parent/n,E/n)
 train=(E/'train.txt').read_text().splitlines();val=(E/'val.txt').read_text().splitlines();assert len(train)==631 and len(set(train))==623 and not set(map(sha,train))&set(map(sha,val))
 registry=sha(B/'active_models.json');weights=B/'yolo11n.pt';model=YOLO(str(weights));assert len(model.names)==80
 plan=json.loads((parent/'locked_plan.json').read_text(encoding='utf-8'));plan.update(multi_scale=1/3,lrf=.1,epochs=45,train_count=631,unique_images=623,initial_sha256=sha(weights),lr0=.001,warmup_epochs=3,registry_sha256=registry,scope='Generic80-class initialization; two-class head adapted during training. Same cleaned data and generic45 optimizer settings; multi_scale1/3 added, compute per batch varies. Baseline historical foodseg reference only.')
 save('locked_plan.json',plan);shutil.copyfile(parent/'old_before.json',E/'old_before.json')
 yaml=E/'data.yaml';yaml.write_text(f'path: {E.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: meat\n  1: vegetable\n')
 save('status.json',dict(stage='training',adopted=False))
 model.add_callback('on_train_epoch_end',lambda trainer: save('observed_train_sizes.json',getattr(trainer,'size_counts',{})))
 model.train(trainer=AuditedTrainer,multi_scale=1/3,data=str(yaml),epochs=45,imgsz=960,batch=8,device=0,workers=0,optimizer='AdamW',lr0=.001,lrf=.1,warmup_epochs=3,seed=42,deterministic=True,project=str(E),name='training',patience=45,plots=False,cache=False,amp=False,mosaic=0,mixup=0,fliplr=0,translate=0,scale=0,hsv_h=0,hsv_s=0,hsv_v=0,save=True)
 model=YOLO(str(E/'training/weights/last.pt'));assert model.names=={0:'meat',1:'vegetable'}
 rows=json.loads((B/'food_presence_patches_v3/reviewed_manifest.json').read_text(encoding='utf-8'))['records'];out=[]
 for r in rows:
  p=model.predict(r['crop'],conf=.25,imgsz=640,rect=False,device=0,verbose=False)[0];out.append(dict(id=r['id'],reference=r['label'],source=r['source_group'],predictions=[dict(category=model.names[int(q.cls[0])],confidence=float(q.conf[0]),box=q.xyxy[0].tolist()) for q in p.boxes]))
 save('after.json',out)
 for name,data in [('old_after',yaml),('aligned_after',B/'food_aligned_replay_v1/aligned.yaml')]:
  m=model.val(data=str(data),imgsz=640,device=0,batch=8,workers=0,plots=False,project=str(E),name=name,verbose=False);save(name+'.json',dict(map50=float(m.box.map50),map50_95=float(m.box.map)))
 assert sha(B/'active_models.json')==registry
 save('status.json',dict(stage='completed',adopted=False,default_unchanged=True))
if __name__=='__main__':main()
