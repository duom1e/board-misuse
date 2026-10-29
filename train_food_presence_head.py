"""Fixed leave-one-video-out pilot; no backbone/VLM fine-tuning or adoption."""
import os,json,hashlib
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'food_presence_head_v1';CLASSES=['background','meat','vegetable']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 E.mkdir(exist_ok=False)
 m=B/'food_presence_patches_v2/reviewed_manifest.json';rows=json.loads(m.read_text(encoding='utf-8'))['records']
 weights=B/'yolo11s.pt';active_sha=sha(B/'active_models.json');sources=sorted({r['source_group'] for r in rows})
 assert len({r['crop_sha256'] for r in rows})==len(rows)
 folds=[]
 for source in sources:
  train=[r for r in rows if r['source_group']!=source];test=[r for r in rows if r['source_group']==source]
  assert {r['label'] for r in train}==set(CLASSES)
  assert not ({r['image_sha256'] for r in train}&{r['image_sha256'] for r in test})
  folds.append(dict(held_source=source,train_ids=[r['id'] for r in train],held_ids=[r['id'] for r in test]))
 for r in rows:
  assert r['training_eligible'] and sha(r['crop'])==r['crop_sha256'] and sha(r['image'])==r['image_sha256']
 plan=dict(manifest_sha256=sha(m),backbone=str(weights),backbone_sha256=sha(weights),classes=CLASSES,seed=20261029,epochs=120,optimizer='Adam',lr=.01,features='Frozen original YOLO11s embedding, L2-normalized, 640 square rect=False',folds=folds,checkpoint='fixed last epoch; no tuning or best fold selection',scope='36 nonexpert manual crops, 4 consumed videos, source-held development pilot; not blind or end-to-end evaluation',active_models_sha256=active_sha)
 save('locked_plan.json',plan)
 import torch
 from ultralytics import YOLO
 torch.set_num_threads(4)
 model=YOLO(str(weights));vectors=model.embed([r['crop'] for r in rows],imgsz=640,rect=False,batch=8,device='cpu',verbose=False)
 x=torch.nn.functional.normalize(torch.stack([v.detach().float().cpu() for v in vectors]),dim=1);y=torch.tensor([CLASSES.index(r['label']) for r in rows]);allresults=[]
 torch.save(dict(features=x,ids=[r['id'] for r in rows],plan_sha256=sha(E/'locked_plan.json')),E/'features.pt')
 for fold in folds:
  torch.manual_seed(plan['seed']);train=torch.tensor([r['source_group']!=fold['held_source'] for r in rows]);head=torch.nn.Linear(x.shape[1],3);opt=torch.optim.Adam(head.parameters(),lr=.01)
  counts=torch.bincount(y[train],minlength=3).float();cw=counts.sum()/(3*counts);losses=[]
  for epoch in range(120):
   opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(x[train]),y[train],weight=cw);loss.backward();opt.step();losses.append(float(loss.detach()))
  with torch.no_grad():pred=head(x).argmax(1)
  held=[dict(id=r['id'],reference=r['label'],prediction=CLASSES[int(pred[i])],source_group=r['source_group']) for i,r in enumerate(rows) if not bool(train[i])]
  torch.save(dict(state_dict=head.state_dict(),embedding_dim=x.shape[1],classes=CLASSES,held_source=fold['held_source'],plan=plan),E/f'{fold["held_source"]}_head_last.pt')
  result=dict(held_source=fold['held_source'],train_count=int(train.sum()),train_matches=int((pred[train]==y[train]).sum()),held_count=len(held),held_matches=sum(r['reference']==r['prediction'] for r in held),rows=held,loss_history=losses)
  allresults.append(result);save('partial_results.json',dict(folds=allresults));print(fold['held_source'],result['held_matches'],'/',len(held),flush=True)
 assert sha(B/'active_models.json')==active_sha
 save('results.json',dict(folds=allresults,adopted=False,backbone_trained=False,vlm_trained=False,head_training_performed=True,active_models_unchanged=True))
if __name__=='__main__':main()
