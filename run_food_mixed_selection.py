"""Prospectively fixed operational checkpoint selection on development data."""
import os,json,hashlib,random,argparse
from pathlib import Path
os.environ['YOLO_CONFIG_DIR']=r'D:\CodexProjects\board-misuse\work\yolo-config'
os.environ['YOLO_AUTOINSTALL']='False'
os.environ['YOLO_OFFLINE']='True'
from ultralytics import YOLO
from run_canteen_food_detector import evaluate
B=Path(__file__).resolve().parent;E=B/'experiments/canteen_food_mixed_operational'

def operational(result):
 rows=result['rows'];classes={}
 for ci,name in enumerate(['meat','vegetable']):
  tp=sum(r['tp'] for r in rows if r['class']==name)
  fn=sum(r['fn'] for r in rows if r['class']==name)
  predicted=sum(p['class']==ci for r in rows for p in r['predictions'])
  fp=predicted-tp
  classes[name]={'tp':tp,'fp':fp,'fn':fn,'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
 c=result['counts'];p=c['precision'] or 0
 empty_fp=sum(r['fp'] for r in rows if r['class']=='empty')
 feasible=p>=.75 and classes['meat']['recall']>=2/3 and classes['vegetable']['recall']>=.5 and empty_fp==0
 macro=sum(x['f1'] for x in classes.values())/2
 return {'feasible':feasible,'classes':classes,'empty_fp':empty_fp,'macro_f1':macro,'rank':[int(feasible),macro,min(x['recall'] for x in classes.values()),p,-c['fp']]}

def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def score_views(result,records):
 score=operational(result);camera={r['image']:r.get('camera','overhead') for r in records};views={}
 for view in sorted(set(camera.values())):
  rs=[r for r in result['rows'] if camera[r['image']]==view]
  counts={k:sum(r[k] for r in rs) for k in ['tp','fp','fn']};counts['precision']=counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None
  v=operational({'rows':rs,'counts':counts});views[view]={'counts':counts,**v}
 score['views']=views
 score['feasible']=score['feasible'] and all(all(c['recall']>=.5 for c in v['classes'].values()) for v in views.values())
 score['rank'][0]=int(score['feasible'])
 return score
def regression(weights,yaml,name):
 m=YOLO(str(weights));r=m.val(data=str(yaml),split='val',imgsz=640,conf=.001,iou=.7,device=0,batch=8,workers=0,plots=False,project=str(E),name=name,verbose=False)
 return {'map50':float(r.box.map50),'map50_95':float(r.box.map),'precision_curve_summary':float(r.box.mp),'recall_curve_summary':float(r.box.mr)}

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='canteen_food_boxes_v2');parser.add_argument('--experiment',default='canteen_food_mixed_operational');parser.add_argument('--require-views',action='store_true');args=parser.parse_args()
 assert Path(args.dataset).name==args.dataset and Path(args.experiment).name==args.experiment
 E=B/'experiments'/args.experiment
 E.mkdir(exist_ok=False);d=E/'data';d.mkdir()
 source=json.loads((B/args.dataset/'manifest.json').read_text(encoding='utf-8'));assert source['review_status']=='assistant_overlay_reviewed_not_expert'
 new=[r for r in source['records'] if r['split']=='train'];val=[r for r in source['records'] if r['split']=='val']
 rng=random.Random(42)
 def sample(file,n):
  ps=[x.strip() for x in (B/'foodseg_two_class'/file).read_text(encoding='utf-8-sig').splitlines() if x.strip()];rng.shuffle(ps);return ps[:n]
 old=sample('train.txt',128);oldval=sample('validation.txt',64)
 # Repeated paths deliberately control exposure, never claim additional unique images.
 train=old+[r['image'] for r in new]*3
 (d/'train.txt').write_text('\n'.join(train)+'\n',encoding='utf-8')
 (d/'val.txt').write_text('\n'.join(r['image'] for r in val)+'\n',encoding='utf-8')
 (d/'oldval.txt').write_text('\n'.join(oldval)+'\n',encoding='utf-8')
 for name,va in [('mixed','val.txt'),('old_regression','oldval.txt')]:
  (d/(name+'.yaml')).write_text(f'path: {d.as_posix()}\ntrain: train.txt\nval: {va}\nnames:\n  0: meat\n  1: vegetable\n',encoding='utf-8')
 weights=B/'training_runs/foodseg_2cls/weights/best.pt'
 registry_hash=hashlib.sha256((B/'active_models.json').read_bytes()).hexdigest()
 plan={'initial_weights':str(weights),'initial_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'dataset_manifest_sha256':hashlib.sha256((B/'canteen_food_boxes_v2/manifest.json').read_bytes()).hexdigest(),'old_training':old,'old_regression_validation':oldval,'unique_train_images':165,'train_entries':239,'real_training_images':37,'real_exposure':3,'epochs':40,'optimizer':'AdamW','lr0':.0002,'seed':42,'imgsz':640,'batch':8,'save_period':10,'candidates':'all periodic epoch*.pt, best.pt, last.pt; stable lexical order ties','selection':'feasible first, macro F1, minimum class recall, precision, fewer FP; only 7-image development validation','gates':{'precision':.75,'meat_recall':2/3,'vegetable_recall':.5,'empty_false_boxes':0},'confidence':.5,'match_iou':.5,'regression_gate':'old 64-image validation mAP50 drop <= .05','post_selection_old8':'evaluate once only if development operational gates pass; NOT independent test','scope':'Same video with adjacent train/development frames, highly correlated. No final independent benchmark. No default promotion.','registry_before_sha256':registry_hash}
 plan.update(dataset=args.dataset,dataset_manifest_sha256=hashlib.sha256((B/args.dataset/'manifest.json').read_bytes()).hexdigest(),unique_train_images=len(set(train)),train_entries=len(train),real_training_images=len(new),validation_images=len(val),selection='feasible first, macro F1, minimum class recall, precision, fewer FP; development validation only',require_views=args.require_views,extra_view_gate='each camera meat and vegetable recall >= .5' if args.require_views else None,dataset_scope=source['scope'])
 write(E/'locked_plan.json',plan)
 if args.require_views:
  reference=json.loads((B/'experiments/canteen_food_mixed_operational/decision.json').read_text())['selected_for_development']['weights']
  ref=evaluate(YOLO(reference),val);write(E/'previous_reference_validation.json',{'weights':reference,'validation':ref,'operational':score_views(ref,val)})
 base_old=regression(weights,d/'old_regression.yaml','old_before');write(E/'old_before.json',base_old)
 model=YOLO(str(weights));model.train(data=str(d/'mixed.yaml'),epochs=40,imgsz=640,batch=8,device=0,workers=0,optimizer='AdamW',lr0=.0002,seed=42,deterministic=True,project=str(E),name='training',patience=40,plots=False,cache=False,amp=True,mosaic=0,mixup=0,fliplr=.5,translate=.05,scale=.1,save_period=10)
 wd=E/'training/weights';candidates=sorted(wd.glob('*.pt'));evaluated=[]
 for p in candidates:
  result=evaluate(YOLO(str(p)),val);score=score_views(result,val) if args.require_views else operational(result)
  evaluated.append({'weights':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'validation':result,'operational':score})
  write(E/'checkpoint_validation.json',evaluated)
 chosen=max(evaluated,key=lambda r:r['operational']['rank'])
 old_after=regression(chosen['weights'],d/'old_regression.yaml','old_after')
 passed=chosen['operational']['feasible'] and old_after['map50']>=base_old['map50']-.05
 development=evaluate(YOLO(chosen['weights']),[r for r in source['records'] if r['split']=='test']) if passed else None
 write(E/'decision.json',{'selected_for_development':chosen,'old_before':base_old,'old_after':old_after,'development_and_regression_gates_passed':passed,'old8_evaluated':development is not None,'old8_development':development,'adopted':False,'reason':'No independent kitchen/action validation; tiny exposed development set.','registry_unchanged':hashlib.sha256((B/'active_models.json').read_bytes()).hexdigest()==registry_hash})
 print(json.dumps({'selected':chosen['weights'],'operational':chosen['operational'],'old_before':base_old,'old_after':old_after,'gates_passed':passed,'old8':development['counts'] if development else None}))
