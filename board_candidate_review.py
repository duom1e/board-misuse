"""Optional uncalibrated research opinions; does not delete boxes or decide compliance."""
import json,math,os
from pathlib import Path
B=Path(__file__).resolve().parent
def review_candidates(image,boxes):
    if not boxes:return []
    os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_CONFIG_DIR']=str(B.parent/'work/yolo-config')
    import torch,numpy as np
    from ultralytics import YOLO
    crops=[]
    for x1,y1,x2,y2 in boxes:
        crop=image.crop((max(0,math.floor(x1)),max(0,math.floor(y1)),min(image.width,math.ceil(x2)),min(image.height,math.ceil(y2))))
        crops.append(np.asarray(crop)[:,:,::-1].copy())
    model=YOLO(str(B/'yolo11s.pt'));vectors=model.embed(crops,imgsz=640,rect=False,batch=8,device=0,verbose=False)
    x=torch.nn.functional.normalize(torch.stack([v.detach().float().cpu() for v in vectors]),dim=1)
    results=[{'box':box,'status':'needs_review','opinions':{},'box_preserved':True} for box in boxes]
    for name,folder in [('small','board_table_rectfixed_v1'),('expanded','board_table_rectfixed_v2')]:
        checkpoint=torch.load(B/'experiments'/folder/'head_last.pt',map_location='cpu',weights_only=True)
        head=torch.nn.Linear(checkpoint['embedding_dim'],2);head.load_state_dict(checkpoint['state_dict'])
        with torch.no_grad():probs=head(x).softmax(1)
        for r,p in zip(results,probs):r['opinions'][name]={'category':['board','table'][int(p.argmax())],'board_score':float(p[0]),'score_is_calibrated':False}
    return results
