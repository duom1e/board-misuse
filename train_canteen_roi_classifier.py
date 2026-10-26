"""Frozen food vision features + trained3-way ROI head, single-scene positive control."""
import os,json,hashlib,random
from pathlib import Path
BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work/yolo-config');os.environ['YOLO_AUTOINSTALL']='False';os.environ['YOLO_OFFLINE']='True'
import numpy as np
import torch
from ultralytics import YOLO
from run_experiment import write_json
from experiment_data import sha256
DATA=BASE/'canteen_roi_food_v1';OUT=BASE/'experiments/canteen_roi_food_v1'
CLASSES=['meat','vegetable','empty']

def metrics(labels,predictions):
 matrix=np.zeros((3,4),dtype=int)
 for y,p in zip(labels,predictions):matrix[y,p if p>=0 else 3]+=1
 stats=[]
 for i,name in enumerate(CLASSES):
  tp=matrix[i,i];fp=matrix[:,i].sum()-tp;fn=matrix[i,:].sum()-tp
  precision=float(tp/(tp+fp)) if tp+fp else 0.;recall=float(tp/(tp+fn)) if tp+fn else 0.
  stats.append({'class':name,'support':int(tp+fn),'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.})
 return {'confusion_rows_truth_columns_meat_vegetable_empty_unknown':matrix.tolist(),'per_class':stats,'accuracy':float(sum(y==p for y,p in zip(labels,predictions))/len(labels)),'macro_f1':sum(x['f1'] for x in stats)/3}

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
 if manifest['review_status']!='assistant_visually_reviewed_all_selected_crops_not_expert':raise RuntimeError('Crops not reviewed')
 registry=json.loads((BASE/'active_models.json').read_text(encoding='utf-8-sig'));weights=registry['foodseg']['weights']
 plan={'scope':'Same-video fixed ROI3-way positive control; not detector accuracy, contact or full misuse ability','classes':CLASSES,'base_weights':weights,'base_sha256':sha256(weights),'manifest_sha256':sha256(DATA/'manifest.json'),'embed_layers':[10],'imgsz':640,'train':'Frozen YOLO features; train-only normalization; balanced3-class linear head; fullbatch AdamW lr.01 weight_decay.1 epochs300 seed42','minimum_confidence':.75,'validation_gate':{'macro_f1':.7,'accuracy':.75,'each_class_recall':.5},'no_hyperparameter_search':True,'limitations':['Single video and camera, correlated frames/near pairs retained; not independent generalization.','Only21 training crops,7 development crops and8 held-out-time crops.','One-class-at-a-time labels; no mixed-food, unfamiliar-food, raw/cooked or OOD detection validation.','Assistant visual labels, not expert ground truth; empty does not certify hygiene.','No default model or registry changes; local noncommercial research only.']}
 write_json(OUT/'locked_plan.json',plan)
 torch.manual_seed(42);random.seed(42);np.random.seed(42);torch.set_num_threads(4)
 rows=manifest['records']
 for r in rows:
  if sha256(r['image'])!=r['sha256']:raise RuntimeError('Image changed')
 base=YOLO(weights)
 # Frozen feature extraction is independent of all ROI labels; no test normalization or fitting.
 embeddings=base.predict([r['image'] for r in rows],embed=[10],imgsz=640,device=0,batch=8,verbose=False)
 x=torch.stack([v.detach().cpu().float().flatten() for v in embeddings]);y=torch.tensor([CLASSES.index(r['class']) for r in rows])
 ids={split:torch.tensor([i for i,r in enumerate(rows) if r['split']==split],dtype=torch.long) for split in ['train','val','test']}
 train=ids['train'];mean=x[train].mean(0);std=x[train].std(0).clamp_min(.01);x=(x-mean)/std
 head=torch.nn.Linear(x.shape[1],3)
 frequency=torch.bincount(y[train],minlength=3).float();loss_fn=torch.nn.CrossEntropyLoss(weight=frequency.sum()/(3*frequency))
 optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.1);losses=[]
 for epoch in range(300):
  optimizer.zero_grad();loss=loss_fn(head(x[train]),y[train]);loss.backward();optimizer.step()
  if epoch%50==0:losses.append({'epoch':epoch+1,'loss':float(loss.detach())})
 head.eval();torch.save({'head_state':head.state_dict(),'mean':mean,'std':std,'classes':CLASSES,'feature_dimensions':x.shape[1],'base_weights':weights,'base_sha256':plan['base_sha256'],'embed_layers':[10],'confidence':.75,'scope':plan['scope']},OUT/'roi_classifier.pt')
 result={'plan':plan,'train_losses':losses,'evaluations':{}}
 def evaluate(split):
  index=ids[split]
  with torch.no_grad():prob=head(x[index]).softmax(1);confidence,pred=prob.max(1)
  labels=y[index].tolist();predicted=[int(p) if float(c)>=.75 else -1 for p,c in zip(pred,confidence)]
  return {'metrics':metrics(labels,predicted),'rows':[{'image':rows[int(i)]['image'],'true':CLASSES[t],'predicted':CLASSES[p] if p>=0 else 'unknown','confidence':float(c),'probabilities':v.tolist()} for i,t,p,c,v in zip(index,labels,predicted,confidence,prob)]}
 result['evaluations']['train']=evaluate('train');result['evaluations']['val']=evaluate('val');m=result['evaluations']['val']['metrics']
 passed=m['macro_f1']>=.7 and m['accuracy']>=.75 and all(row['recall']>=.5 for row in m['per_class'])
 result['validation_gate_passed']=passed;result['held_out_time_test_evaluated']=False
 if passed:result['evaluations']['test']=evaluate('test');result['held_out_time_test_evaluated']=True
 if sha256(DATA/'manifest.json')!=plan['manifest_sha256']:raise RuntimeError('Labels changed')
 write_json(OUT/'results.json',result)
 print(json.dumps({'feature_dimensions':x.shape[1],'split_counts':{k:len(v) for k,v in ids.items()},'validation_passed':passed,'metrics':{k:v['metrics'] for k,v in result['evaluations'].items()}},ensure_ascii=False))

if __name__=='__main__':main()
