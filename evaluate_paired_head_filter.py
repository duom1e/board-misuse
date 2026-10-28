"""Evaluate prelocked experimental heads on frozen 4B box proposals. No deployment."""
import os,json,hashlib,math
from pathlib import Path
from PIL import Image
B=Path(__file__).resolve().parent;E=B/'experiments/paired_video_head_filter'
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_CONFIG_DIR']=str(B.parent/'work/yolo-config')
def main():
    assert not (E/'results.json').exists()
    plan=json.loads((E/'locked_plan.json').read_text(encoding='utf-8'));pred=json.loads((B/'experiments/qwen4b_paired_video_v1/results.json').read_text(encoding='utf-8'))['rows'];labels={r['case']:r for r in json.loads((B/'paired_video_validation_v1/manifest.json').read_text(encoding='utf-8'))['records']};crops=[]
    for r in pred:
        if 'pixel_boxes' not in r:continue
        im=Image.open(labels[r['case']]['image']).convert('RGB')
        for i,box in enumerate(r['pixel_boxes']['boards']):
            x1,y1,x2,y2=box;box=[max(0,math.floor(x1)),max(0,math.floor(y1)),min(im.width,math.ceil(x2)),min(im.height,math.ceil(y2))]
            if box[2]<=box[0] or box[3]<=box[1]:continue
            path=E/f"{r['case']}_{i}.jpg";im.crop(box).save(path,quality=95);crops.append({'case':r['case'],'image':str(path),'box':box})
    import torch
    from ultralytics import YOLO
    model=YOLO(str(B/'yolo11s.pt'));features=model.embed([r['image'] for r in crops],imgsz=640,rect=False,batch=8,device=0,verbose=False);x=torch.nn.functional.normalize(torch.stack([v.detach().float().cpu() for v in features]),dim=1);results={}
    for name,meta in plan['heads'].items():
        path=Path(meta['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256'];ckpt=torch.load(path,map_location='cpu',weights_only=True);head=torch.nn.Linear(ckpt['embedding_dim'],2);head.load_state_dict(ckpt['state_dict'])
        with torch.no_grad():prob=head(x).softmax(1)
        scored=[{**r,'predicted':['board','table'][int(p.argmax())],'board_probability':float(p[0])} for r,p in zip(crops,prob)];cases=[]
        for r in pred:
            cases.append({'case':r['case'],'expected':labels[r['case']]['expected_board'],'predicted':'unknown' if 'pixel_boxes' not in r else 'present' if any(c['case']==r['case'] and c['predicted']=='board' for c in scored) else 'absent','status':'needs_review'})
        results[name]={'crops':scored,'presence_diagnostic':cases}
    (E/'results.json').write_text(json.dumps({'results':results,'adopted':False,'training_performed':False},indent=2),encoding='utf-8');print({n:[(r['case'],r['expected'],r['predicted']) for r in d['presence_diagnostic']] for n,d in results.items()})
if __name__=='__main__':main()
