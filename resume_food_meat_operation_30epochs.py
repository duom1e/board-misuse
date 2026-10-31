import os,json,hashlib
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
from ultralytics import YOLO
B=Path(__file__).resolve().parent;E=B/'food_meat_operation_30epochs_v1'
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
model=YOLO(str(E/'training/weights/last.pt'))
model.train(resume=True)
model=YOLO(str(E/'training/weights/last.pt'))
rows=json.loads((E/'before.json').read_text(encoding='utf-8'));out=[]
refs={r['id']:r for r in json.loads((B/'food_presence_patches_v3/reviewed_manifest.json').read_text(encoding='utf-8'))['records']}
for r in rows:
 p=model.predict(refs[r['id']]['crop'],conf=.25,imgsz=640,rect=False,device=0,verbose=False)[0]
 out.append({**r,'predictions':[dict(category=model.names[int(q.cls[0])],confidence=float(q.conf[0]),box=q.xyxy[0].tolist()) for q in p.boxes]})
save('after.json',out)
for name,data in [('old_after',E/'data.yaml'),('aligned_after',B/'food_aligned_replay_v1/aligned.yaml')]:
 m=model.val(data=str(data),imgsz=640,device=0,batch=8,workers=0,plots=False,project=str(E),name=name,verbose=False)
 save(name+'.json',dict(map50=float(m.box.map50),map50_95=float(m.box.map)))
plan=json.loads((E/'locked_plan.json').read_text(encoding='utf-8'))
assert hashlib.sha256((B/'active_models.json').read_bytes()).hexdigest()==plan['registry_sha256']
save('status.json',dict(stage='completed',adopted=False,default_unchanged=True,resumed_after_interruption=True))
