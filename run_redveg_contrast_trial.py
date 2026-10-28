"""Locked mixed training with new source contrasts and retained old ability."""
import json,hashlib,random,argparse,shutil
from pathlib import Path
import run_food_mixed_selection as selection
from run_canteen_food_detector import evaluate
from ultralytics import YOLO
import cv2
B=Path(__file__).resolve().parent;E=B/'experiments/redveg_contrast_frozen_backbone'
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def category_summary(result):
 return {'meat_predictions':sum(p['class']==0 for r in result['rows'] for p in r['predictions']),
 'vegetable_predictions':sum(p['class']==1 for r in result['rows'] for p in r['predictions']),
 'vegetable_images_with_vegetable_prediction':sum(r['class']=='vegetable' and any(p['class']==1 for p in r['predictions']) for r in result['rows']),**result['counts']}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--freeze',type=int,default=10);parser.add_argument('--experiment',default='redveg_contrast_frozen_backbone');parser.add_argument('--minced-shapes',action='store_true');parser.add_argument('--minced-repeat',type=int,default=3);args=parser.parse_args()
 assert args.minced_repeat in (3,12)
 assert args.freeze in (0,10) and Path(args.experiment).name==args.experiment
 E=B/'experiments'/args.experiment
 E.mkdir(exist_ok=False);D=E/'data';D.mkdir();selection.E=E
 hf=json.loads((B/'canteen_food_multiview_v1/manifest.json').read_text(encoding='utf-8'))
 new=json.loads((B/'redveg_meat_contrast_v1/manifest.json').read_text(encoding='utf-8'))
 probe=json.loads((B/'bilibili_prep_manual_v1/manifest.json').read_text(encoding='utf-8'))
 assert hf['review_status']=='assistant_overlay_reviewed_not_expert' and new['review_status']=='assistant_overlay_reviewed_not_expert'
 assert probe['review_status']=='assistant_food_overlay_reviewed_not_expert'
 old=[p.strip() for p in (B/'foodseg_two_class/train.txt').read_text(encoding='utf-8-sig').splitlines() if p.strip()];random.Random(42).shuffle(old);old=old[:512]
 previous=json.loads((B/'experiments/canteen_food_mixed_operational/locked_plan.json').read_text())
 oldval=previous['old_regression_validation'];val=[r for r in hf['records'] if r['split']=='val'];real=[r for r in hf['records'] if r['split']=='train']
 train=old+[r['image'] for r in real]+[r['image'] for r in new['records']]*3
 shapes=None;reserved=[]
 if args.minced_shapes:
  assert args.freeze==0
  shapes=json.loads((B/'minced_shape_annotation_v1/manifest.json').read_text(encoding='utf-8'));assert shapes['review_status']=='assistant_overlay_reviewed_not_expert'
  extra=[r for r in shapes['records'] if r['intended_role']=='training_candidate']
  reserved=[r for r in shapes['records'] if r['intended_role'].startswith('reserved_') and r['localization_uncertainty']=='normal']
  assert len(extra)==11 and len(reserved)==5
  assert not ({r['source_group'] for r in extra}&{r['source_group'] for r in reserved})
  staged=[]
  for r in extra:
   r=dict(r);ip=D/'images/train'/Path(r['image']).name;lp=D/'labels/train'/(ip.stem+'.txt');ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(r['image'],ip);shutil.copyfile(r['label'],lp);assert digest(ip)==r['sha256'] and digest(lp)==r['label_sha256'];r.update(image=str(ip),label=str(lp));staged.append(r)
  extra=staged
  train += [r['image'] for r in extra]*args.minced_repeat
 held=[dict(r) for r in probe['records'] if r['localization_uncertainty']=='normal']
 source=json.loads((B/'bilibili_prep_screening/manifest.json').read_text(encoding='utf-8'))['source'];cap=cv2.VideoCapture(source);fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
 for r in held:r['seconds']=r['frame']/fps
 for name,paths in [('train',train),('val',[r['image'] for r in val]),('oldval',oldval)]:
  (D/(name+'.txt')).write_text('\n'.join(paths)+'\n',encoding='utf-8')
 for name,validation in [('mixed','val.txt'),('old_regression','oldval.txt')]:
  (D/(name+'.yaml')).write_text(f'path: {D.as_posix()}\ntrain: train.txt\nval: {validation}\nnames:\n  0: meat\n  1: vegetable\n',encoding='utf-8')
 weights=B/'training_runs/foodseg_2cls/weights/best.pt';registry=digest(B/'active_models.json')
 plan={'initial_weights':str(weights),'initial_sha256':digest(weights),'old_training':old,'old_regression_validation':oldval,'hf_training_images':52,'contrast_training_images':16,'contrast_exposure':3,'unique_training_images':len(set(train)),'training_entries':len(train),'epochs':30,'freeze':10,'optimizer':'AdamW','lr0':.0001,'batch':8,'imgsz':640,'seed':42,'save_period':10,'checkpoint_selection':'Same 15 HF development images, score_views fixed operational gates and ranking, lexically first tie. All epoch*.pt/best/last candidates.','validation_gates':{'precision':.75,'meat_recall':2/3,'vegetable_recall':.5,'each_camera_each_class_recall':.5,'no_bulk_food_false_boxes':0},'regression_gate':'mAP50 on SAME old64 validation drops at most .05','bilibili20_diagnostic':'Evaluate selected checkpoint once regardless of gate; never used for checkpoint choice. Exposed development probe, not independent final test.','bilibili20_success_gate':'TP >=6/18, meat predictions <=4, precision >=.5; no deployment decision even if passes','confidence':.5,'match_iou':.5,'manifests_sha256':{p:digest(B/p/'manifest.json') for p in ['canteen_food_multiview_v1','redveg_meat_contrast_v1','bilibili_prep_manual_v1']},'scope':'Changes data coverage, exposure, optimizer schedule and freeze together; not a single-variable causal ablation. New contrast sources one per class; same-source HF near-frame validation; no general deployment claim.','registry_before_sha256':registry}
 write(E/'locked_plan.json',plan)
 if args.freeze==0:
  ref=json.loads((B/'experiments/redveg_contrast_frozen_backbone/locked_plan.json').read_text(encoding='utf-8'))
  for key in ['initial_sha256','old_training','old_regression_validation','manifests_sha256','epochs','optimizer','lr0','batch','imgsz','seed','contrast_exposure']+([] if args.minced_shapes else ['training_entries','unique_training_images']):
   assert plan[key]==ref[key],key
  plan.update(freeze=0,comparison_reference='redveg_contrast_frozen_backbone',comparison='Same data, settings, selection and initial weights; only freeze10 versus freeze0 changes. Both exposed development diagnostics, not independent test.')
  write(E/'locked_plan.json',plan)
 if shapes:
  plan.update(minced_manifest_sha256=digest(B/'minced_shape_annotation_v1/manifest.json'),extra_training_images=11,extra_exposure=args.minced_repeat,reserved_source_images=5,reserved_source_gt=sum(len(r['boxes_xyxy']) for r in reserved),reserved_release='Only after operational and old-domain gates pass. Compare selected candidate and previous unfrozen candidate once, no checkpoint/threshold tuning.',reserved_success='vegetable recall >=.5, precision >=.75, meat predictions0; no meat/empty samples so not complete benchmark',comparison_reference='redveg_contrast_unfrozen_control',comparison='Same unfrozen settings with11 additional training images; more exposure adds batches/updates, not equal-compute ablation.')
  if args.minced_repeat==12:
   ref=json.loads((B/'experiments/minced_shape_adaptation_v1_reviewed/locked_plan.json').read_text())
   for k in ['initial_sha256','old_training','old_regression_validation','minced_manifest_sha256','epochs','optimizer','lr0','batch','imgsz','seed','freeze','manifests_sha256','validation_gates']:
    assert plan[k]==ref[k],k
   plan.update(comparison_reference='minced_shape_adaptation_v1_reviewed',comparison='New11 image exposure3 to12, same unique data/initial weights/settings/selection; extra batches and optimizer steps, not equal-compute ablation.',fit_release_gate='Additional training-fit gate: precision/recall >=.9 and each class recall>=.8; selected weights fixed before fit check, no re-selection.')
  write(E/'locked_plan.json',plan)
 # Exact hash audit across all training and all development inputs.
 unique=set(train);hashes={p:digest(p) for p in unique};train_hashes=set(hashes.values());leaks=[r['image'] for r in val+held+reserved if digest(r['image']) in train_hashes]+[p for p in oldval if digest(p) in train_hashes];assert not leaks
 for p in unique:
  lp=Path(p.replace('images','labels')).with_suffix('.txt');assert lp.exists(),lp
  for line in lp.read_text().splitlines():
   c,x,y,w,h=map(float,line.split());assert c in (0,1) and 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
 write(E/'data_audit.json',{'unique':len(unique),'entries':len(train),'exact_train_development_duplicates':leaks,'same_video_correlation':'HF known; contrast video sources kept outside B diagnostic source','new_labels_review':'assistant not expert'})
 old_before=selection.regression(weights,D/'old_regression.yaml','old_before');write(E/'old_before.json',old_before)
 model=YOLO(str(weights));model.train(data=str(D/'mixed.yaml'),epochs=30,freeze=args.freeze,imgsz=640,batch=8,device=0,workers=0,optimizer='AdamW',lr0=.0001,seed=42,deterministic=True,project=str(E),name='training',patience=30,plots=False,cache=False,amp=True,mosaic=0,mixup=0,fliplr=.5,translate=.05,scale=.1,save_period=10)
 rows=[]
 for p in sorted((E/'training/weights').glob('*.pt')):
  r=evaluate(YOLO(str(p)),val);rows.append({'weights':str(p),'sha256':digest(p),'validation':r,'operational':selection.score_views(r,val)});write(E/'checkpoint_validation.json',rows)
 chosen=max(rows,key=lambda r:r['operational']['rank']);old_after=selection.regression(chosen['weights'],D/'old_regression.yaml','old_after')
 result=evaluate(YOLO(chosen['weights']),held);summary=category_summary(result)
 gate=summary['tp']>=6 and summary['meat_predictions']<=4 and (summary['precision'] or 0)>=.5
 write(E/'decision.json',{'selected':chosen,'old_before':old_before,'old_after':old_after,'operational_gate':chosen['operational']['feasible'],'regression_gate':old_after['map50']>=old_before['map50']-.05,'bilibili20':result,'bilibili20_summary':summary,'bilibili20_gate':gate,'adopted':False,'registry_unchanged':digest(B/'active_models.json')==registry,'reason':'Exposed development data, tiny source coverage, no independent final validation.'})
 if shapes:
  decision=json.loads((E/'decision.json').read_text())
  release=decision['operational_gate'] and decision['regression_gate']
  fit=None
  if args.minced_repeat==12:
   fit=evaluate(YOLO(chosen['weights']),extra);fs=selection.operational(fit);fit_gate=(fit['counts']['precision'] or 0)>=.9 and fit['counts']['recall']>=.9 and all(r['recall']>=.8 for r in fs['classes'].values())
   write(E/'extra_training_fit.json',{'result':fit,'classes':fs['classes'],'fit_gate':fit_gate,'scope':'Training fit only, not used to reselect weights.'})
   release=release and fit_gate
  evaluations={}
  if release:
   previous=json.loads((B/'experiments/redveg_contrast_unfrozen_control/decision.json').read_text())['selected']['weights']
   for name,weight in [('previous',previous),('candidate',chosen['weights'])]:
    r=evaluate(YOLO(str(weight)),reserved);evaluations[name]={'result':r,'summary':category_summary(r)}
  write(E/'reserved_source_decision.json',{'released':release,'results':evaluations,'fit_gate_used':args.minced_repeat==12,'scope':'5 images6vegetable groups from one held-out source. Inspected annotations but not predicted before release. No meat/empty cases, not full independent kitchen benchmark.','adopted':False})
 print(json.dumps({'selected':chosen['weights'],'bilibili20':summary,'old_before':old_before,'old_after':old_after,'bilibili_gate':gate}))
