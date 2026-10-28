"""Fixed local detector experiment; consumed video is training, never holdout."""
import json,os,shutil,hashlib
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
os.environ['YOLO_CONFIG_DIR']='D:/CodexProjects/board-misuse/work/yolo-config'
B=Path(__file__).resolve().parent;E=B/'kitchen_cutting_trial_v1';S=B/'kitchen_cutting_batch_v1'
def save(name,obj):(E/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    from ultralytics import YOLO
    E.mkdir(exist_ok=True);assert not (E/'locked_plan.json').exists(),'Already started; inspect status, do not restart'
    cfg=json.loads((B/'active_models.json').read_text(encoding='utf-8'))['board'];old=B/'board_dataset_v2'
    records=json.loads((S/'board_labels_v1/manifest.json').read_text(encoding='utf-8'))['records']
    boxes={r['seconds']:r['box_xyxy'] for r in records};boxes.update({300:[426,0,640,127],390:[452,254,640,360]})
    save('locked_plan.json',dict(epochs=10,seed=42,weights=cfg['weights'],train_video='BV1dHckzwEBA',new_images=10,old_train=str(old/'train.txt'),regression=str(old/'test.txt'),selection='fixed last epoch; no tuning',new_video_usage='training fit only; labels nonexpert, two no-visible-board negatives',rights='Public source permission unverified; local experiment only, do not redistribute media/weights',adopted=False))
    for d in ('images','labels'):(E/d).mkdir(exist_ok=True)
    new=[]
    for t in sorted(list(boxes)+[420,600]):
        p=E/'images'/f'{t}.jpg';shutil.copy2(S/f'{t}.jpg',p);new.append(str(p));box=boxes.get(t)
        text=''
        if box:
            a,b,c,d=box;text=f'0 {(a+c)/1280} {(b+d)/720} {(c-a)/640} {(d-b)/360}\n'
        (E/'labels'/f'{t}.txt').write_text(text,encoding='utf-8')
    prior=(old/'train.txt').read_text(encoding='utf-8').splitlines();prior=[str((old/p).resolve()) if not Path(p).is_absolute() else p for p in prior if p.strip()]
    (E/'train.txt').write_text('\n'.join(prior+new)+'\n',encoding='utf-8')
    (E/'data.yaml').write_text(f'path: {E.as_posix()}\ntrain: train.txt\nval: {(old/"validation.txt").as_posix()}\ntest: {(old/"test.txt").as_posix()}\nnames:\n  0: cutting_board\n',encoding='utf-8')
    save('status.json',dict(state='training',old_images=len(prior),new_images=len(new)))
    model=YOLO(cfg['weights']);model.train(data=str(E/'data.yaml'),epochs=10,batch=8,imgsz=640,device=0,workers=0,seed=42,deterministic=True,amp=False,patience=0,plots=False,save=True,project=str(E),name='train',exist_ok=False)
    candidate=E/'train/weights/last.pt';evaluations={}
    for name,w in [('baseline',cfg['weights']),('candidate',str(candidate))]:
        m=YOLO(w);v=m.val(data=str(old/'data.yaml'),split='test',imgsz=640,batch=8,device=0,workers=0,plots=False,project=str(E),name=name+'_regression');fit=[]
        for t in sorted(list(boxes)+[420,600]):
            pred=m.predict(str(E/'images'/f'{t}.jpg'),conf=.25,imgsz=640,rect=False,verbose=False)[0];out=[]
            for q in pred.boxes:
                xy=q.xyxy[0].tolist();iou=None
                if t in boxes:
                    a,b,c,d=boxes[t];x,y,z,k=xy;inter=max(0,min(c,z)-max(a,x))*max(0,min(d,k)-max(b,y));iou=inter/((c-a)*(d-b)+(z-x)*(k-y)-inter)
                out.append(dict(box=xy,conf=float(q.conf[0]),iou=iou))
            fit.append(dict(seconds=t,expected=boxes.get(t),predictions=out,matched=any((x['iou'] or 0)>=.5 for x in out)))
        evaluations[name]=dict(regression=v.results_dict,training_fit=fit)
    save('results.json',dict(evaluations=evaluations,training_performed=True,adopted=False,independent_new_video_test=False));save('status.json',dict(state='completed',adopted=False));print('COMPLETED',flush=True)
if __name__=='__main__':main()
