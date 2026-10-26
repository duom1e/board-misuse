"""Bounded same-camera positive control, not model promotion."""
import os,json,hashlib,argparse
from pathlib import Path
os.environ['YOLO_CONFIG_DIR']=r'D:\CodexProjects\board-misuse\work\yolo-config'
os.environ['YOLO_AUTOINSTALL']='False'
os.environ['YOLO_OFFLINE']='True'
from ultralytics import YOLO
B=Path(__file__).resolve().parent
E=B/'experiments/canteen_food_boxes_v1'

def evaluate(model,rows):
 result=[]
 for r in rows:
  pred=model.predict(r['image'],conf=.5,iou=.5,imgsz=640,device=0,verbose=False)[0]
  ps=[{'class':int(k),'confidence':float(c),'xyxy':list(map(float,b))} for k,c,b in zip(pred.boxes.cls.cpu(),pred.boxes.conf.cpu(),pred.boxes.xyxy.cpu())]
  used=set();tp=0
  for gt in r['boxes_xyxy']:
   matches=[]
   for j,p in enumerate(ps):
    if j in used or p['class']!=(0 if r['class']=='meat' else 1):continue
    a,b,c,d=gt;x,y,z,t=p['xyxy'];inter=max(0,min(c,z)-max(a,x))*max(0,min(d,t)-max(b,y));union=(c-a)*(d-b)+(z-x)*(t-y)-inter
    if inter/union>=.5:matches.append((inter/union,j))
   if matches:used.add(max(matches)[1]);tp+=1
  result.append({'image':r['image'],'class':r['class'],'seconds':r['seconds'],'tp':tp,'fp':len(ps)-tp,'fn':len(r['boxes_xyxy'])-tp,'predictions':ps})
 sums={k:sum(r[k] for r in result) for k in ['tp','fp','fn']}
 sums['precision']=sums['tp']/(sums['tp']+sums['fp']) if sums['tp']+sums['fp'] else None
 sums['recall']=sums['tp']/(sums['tp']+sums['fn']) if sums['tp']+sums['fn'] else None
 return {'counts':sums,'rows':result}

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--dataset',default='canteen_food_boxes_v1')
 parser.add_argument('--experiment',default='canteen_food_boxes_v1')
 args=parser.parse_args()
 assert Path(args.dataset).name==args.dataset and Path(args.experiment).name==args.experiment
 E=B/'experiments'/args.experiment
 E.mkdir(exist_ok=False)
 data=json.loads((B/args.dataset/'manifest.json').read_text(encoding='utf-8'))
 assert data['review_status']=='assistant_overlay_reviewed_not_expert'
 weights=B/'training_runs/foodseg_2cls/weights/best.pt'
 plan={'base_weights':str(weights),'base_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'epochs':40,'imgsz':640,'batch':8,'optimizer':'AdamW','lr0':.0002,'seed':42,'conf':.5,'match_iou':.5,'scope':'Same-video fixed-ROI board-food-group detector positive control. Off-board foods unlabelled by design. Not full-frame food detector; no independent test; old 8-image test already exposed.','promotion':False,'no_hyperparameter_search':True,'no_threshold_search':True,'no_general_dataset_regression':True}
 plan.update(dataset=args.dataset,dataset_manifest_sha256=hashlib.sha256((B/args.dataset/'manifest.json').read_bytes()).hexdigest(),dataset_scope=data['scope'])
 (E/'locked_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
 rows=data['records'];baseline=YOLO(str(weights))
 before={s:evaluate(baseline,[r for r in rows if r['split']==s]) for s in ['val','test']}
 (E/'baseline.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
 baseline.train(data=str(B/args.dataset/'data.yaml'),epochs=40,imgsz=640,batch=8,device=0,workers=0,optimizer='AdamW',lr0=.0002,seed=42,deterministic=True,project=str(E),name='training',exist_ok=False,patience=40,plots=False,cache=False,amp=True,mosaic=0,mixup=0,fliplr=.5,translate=.05,scale=.1,save=True)
 candidate=YOLO(str(E/'training/weights/best.pt'))
 after={s:evaluate(candidate,[r for r in rows if r['split']==s]) for s in ['val','test']}
 (E/'comparison.json').write_text(json.dumps({'before':before,'after':after,'adopted':False,'scope':plan['scope']},indent=2),encoding='utf-8')
 print(json.dumps({s:{'before':before[s]['counts'],'after':after[s]['counts']} for s in before}))
