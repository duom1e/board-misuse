"""Post-training fixed crop diagnostic, not checkpoint/threshold selection."""
import os,json,hashlib
from pathlib import Path
from PIL import Image
B=Path(__file__).resolve().parent;E=B/'experiments/board_table_video_crop_diagnostic'
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False';os.environ['YOLO_CONFIG_DIR']=str(B.parent/'work/yolo-config')
def main():
    E.mkdir(exist_ok=True);assert not (E/'locked_plan.json').exists()
    regions=json.loads((B/'food_region_relation_v1/manifest.json').read_text(encoding='utf-8'))['records'];rows=[]
    for r in regions:
        p=E/(r['case']+'.jpg');im=Image.open(r['image']);im.crop(r['board']).save(p,quality=95)
        rows.append({'image':str(p),'case':r['case'],'expected':'board','source_image':r['image'],'crop':r['board'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    wood=json.loads((B/'region_new_sources_v1/manifest.json').read_text(encoding='utf-8'))['records']
    for r in wood:
        if r['role']=='wood_table_negative':rows.append({'image':r['image'],'case':r['case'],'expected':'table','sha256':r['sha256']})
    paths={n:B/'experiments'/p/'head_last.pt' for n,p in [('previous','board_table_linear_pilot_v1'),('expanded','board_table_linear_expansion_v2')]}
    plan={'rows':rows,'models':{n:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in paths.items()},'scope':'Post-training diagnostic only. Four human coarse board crops from consumed videos, three previously exposed full wood-table images. Not fresh independent test; no selection/threshold tuning.'}
    (E/'locked_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    import torch
    from ultralytics import YOLO
    model=YOLO(str(B/'yolo11s.pt'));vectors=model.embed([r['image'] for r in rows],imgsz=640,batch=7,device=0,verbose=False);x=torch.nn.functional.normalize(torch.stack([v.detach().float().cpu() for v in vectors]),dim=1);results={}
    for name,p in paths.items():
        ckpt=torch.load(p,map_location='cpu',weights_only=True);head=torch.nn.Linear(ckpt['embedding_dim'],2);head.load_state_dict(ckpt['state_dict'])
        with torch.no_grad():probs=head(x).softmax(1)
        results[name]=[{**r,'predicted':['board','table'][int(v.argmax())],'probabilities':v.tolist()} for r,v in zip(rows,probs)]
    (E/'results.json').write_text(json.dumps({'results':results,'adopted':False},indent=2),encoding='utf-8')
    print({n:[(r['case'],r['expected'],r['predicted']) for r in rows] for n,rows in results.items()})
if __name__=='__main__':main()
