"""Fixed multisource detector experiment; held-out videos never enter training."""
import os,json,hashlib,shutil
from pathlib import Path
from PIL import Image
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'multisource_board_trial_v1'
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 from ultralytics import YOLO
 E.mkdir(exist_ok=True);assert not (E/'locked_plan.json').exists(),'Already started, inspect existing status'
 old=B/'board_dataset_v2';base=json.loads((B/'active_models.json').read_text(encoding='utf-8'))['board']['weights'];examples=[]
 first=json.loads((B/'kitchen_cutting_trial_v1/results.json').read_text(encoding='utf-8'))['evaluations']['baseline']['training_fit']
 for x in first:examples.append(dict(image=str(B/'kitchen_cutting_trial_v1/images'/f"{x['seconds']}.jpg"),box=x['expected'],group='BV1dHckzwEBA'))
 for x in json.loads((B/'kitchen_positive_sources_v1/locked_boxes.json').read_text(encoding='utf-8'))['records']:examples.append(dict(image=x['image'],box=x['box'],group=x['id']))
 for x in json.loads((B/'kitchen_multisource_v1/negative_plan.json').read_text(encoding='utf-8'))['frames']:examples.append(dict(image=x['image'],box=None,group='BV1eB4y1J7B6'))
 hold=json.loads((B/'round_board_sources_v1/locked_boxes.json').read_text(encoding='utf-8'))['records'];assert not {x['group'] for x in examples}&{x['id'] for x in hold}
 save('locked_plan.json',dict(new_train=examples,heldout=hold,epochs=10,optimizer='AdamW',lr=.0003,freeze=0,mosaic=0,warmup_epochs=0,seed=42,selection='fixed last.pt, no epoch selection',scope='Held-out two videos excluded from training but previously consumed diagnostic, not fresh blind test. Nonexpert boxes, rights unverified, local research only.',active_models_sha256=hashlib.sha256((B/'active_models.json').read_bytes()).hexdigest()))
 for d in ('images','labels'):(E/d).mkdir(exist_ok=True)
 paths=[]
 for i,x in enumerate(examples):
  p=E/'images'/f'{i:03}.jpg';shutil.copy2(x['image'],p);w,h=Image.open(p).size;box=x['box'];text=''
  if box:
   a,b,c,d=box;assert 0<=a<c<=w and 0<=b<d<=h;text=f'0 {(a+c)/2/w} {(b+d)/2/h} {(c-a)/w} {(d-b)/h}\n'
  (E/'labels'/f'{i:03}.txt').write_text(text,encoding='utf-8');paths.append(str(p))
 prior=[str((old/x).resolve()) if not Path(x).is_absolute() else x for x in (old/'train.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
 (E/'train.txt').write_text('\n'.join(prior+paths)+'\n',encoding='utf-8');(E/'data.yaml').write_text(f'path: {E.as_posix()}\ntrain: train.txt\nval: {(old/"validation.txt").as_posix()}\ntest: {(old/"test.txt").as_posix()}\nnames:\n  0: cutting_board\n',encoding='utf-8')
 save('status.json',dict(stage='training',old_train=len(prior),new_train=len(paths)))
 m=YOLO(base);m.train(data=str(E/'data.yaml'),epochs=10,optimizer='AdamW',lr0=.0003,lrf=1,warmup_epochs=0,freeze=0,mosaic=0,close_mosaic=0,batch=8,imgsz=640,device=0,workers=0,amp=False,seed=42,patience=0,plots=False,project=str(E),name='train')
 result={}
 for name,w in [('baseline',base),('candidate',str(E/'train/weights/last.pt'))]:
  m=YOLO(w);reg=m.val(data=str(old/'data.yaml'),split='test',batch=8,imgsz=640,device=0,workers=0,plots=False,project=str(E),name=name+'_regression');groups={}
  for label,cases in [('new_train_fit',examples),('heldout_consumed_diagnostic',hold)]:
   rows=[]
   for x in cases:
    box=x.get('box');pred=[]
    for q in m.predict(x['image'],imgsz=640,rect=False,conf=.25,verbose=False)[0].boxes:
     z=q.xyxy[0].tolist();iou=0
     if box:
      a,b,c,d=box;v,y,f,k=z;inter=max(0,min(c,f)-max(a,v))*max(0,min(d,k)-max(b,y));iou=inter/((c-a)*(d-b)+(f-v)*(k-y)-inter)
     pred.append(dict(box=z,iou=iou,confidence=float(q.conf[0])))
    rows.append(dict(image=x['image'],expected=box,predictions=pred,matched=bool(box) and any(q['iou']>=.5 for q in pred)))
   groups[label]=rows
  result[name]=dict(regression=reg.results_dict,**groups)
 save('results.json',dict(models=result,training_performed=True,adopted=False));save('status.json',dict(stage='completed',adopted=False));print('COMPLETED',flush=True)
if __name__=='__main__':main()
