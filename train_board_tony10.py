import os,json,hashlib
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
from ultralytics import YOLO
def main():
    B=Path(__file__).resolve().parent
    O=B/'experiments/board_tony_train10_v1';O.mkdir(exist_ok=False)
    D=B/'experiments/board_tony_mix_v1'
    initial=B/'training_runs/board_v2_real/weights/best.pt'
    records=json.loads((D/'manifest.json').read_text(encoding='utf-8'))['records']
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    train=[Path(p) for p in (D/'train.txt').read_text().splitlines() if p]
    val=[Path(p) for p in (D/'val.txt').read_text().splitlines() if p]
    assert not {sha(p) for p in train}&{sha(p) for p in val}
    for p in train+val:
     assert p.is_file()
     assert Path(str(p).replace('images','labels')).with_suffix('.txt').is_file()
    (O/'plan.json').write_text(json.dumps({'epochs':10,'multi_scale':1/3,'initial':str(initial),'initial_sha256':sha(initial),'train_entries':len(train),'train_unique':len(set(map(sha,train))),'val':len(val),'selection':'fixed last, no best-epoch selection','limits':'3 new training frames and 3 same-source validation frames; previously viewed development data, not independent deployment evaluation'},indent=2),encoding='utf-8')
    def evaluate(weight,tag):
     model=YOLO(str(weight)); result=[]
     for r in records:
      pred=model.predict(r['image'],imgsz=960,rect=False,conf=.25,device=0,verbose=False)[0]
      gt=r['box']; boxes=pred.boxes.xyxy.cpu().tolist()
      def iou(b):
       inter=max(0,min(b[2],gt[2])-max(b[0],gt[0]))*max(0,min(b[3],gt[3])-max(b[1],gt[1]))
       return inter/((b[2]-b[0])*(b[3]-b[1])+(gt[2]-gt[0])*(gt[3]-gt[1])-inter)
      result.append({'image':r['image'],'split':r['split'],'boxes':boxes,'best_iou':max(map(iou,boxes),default=0)})
     (O/(tag+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
     metrics=model.val(data=str(B/'board_dataset/data.yaml'),split='test',workers=0,imgsz=640,device=0,plots=False,verbose=False,project=str(O),name=tag+'_oldtest')
     return {'train_hits':sum(r['best_iou']>=.5 for r in result if r['split']=='train'),'val_hits':sum(r['best_iou']>=.5 for r in result if r['split']=='val'),'old_test_map50':float(metrics.box.map50)}
    before=evaluate(initial,'before')
    YOLO(str(initial)).train(data=str(D/'data.yaml'),epochs=10,imgsz=960,multi_scale=1/3,batch=8,device=0,workers=0,optimizer='AdamW',lr0=.0001,lrf=.1,warmup_epochs=0,patience=0,amp=False,seed=42,mosaic=0,mixup=0,copy_paste=0,degrees=0,translate=0,scale=0,fliplr=0,hsv_h=0,hsv_s=0,hsv_v=0,project=str(O),name='training',exist_ok=False,plots=False)
    after=evaluate(O/'training/weights/last.pt','after')
    (O/'summary.json').write_text(json.dumps({'before':before,'after':after},indent=2),encoding='utf-8')
    print('FINAL',before,after,flush=True)

if __name__ == '__main__':
    main()
