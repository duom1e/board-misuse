"""Train only a linear head over frozen original YOLO features; tiny development pilot."""
import os,json,hashlib,random
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;os.environ['YOLO_CONFIG_DIR']=str(B.parent/'work/yolo-config')
E=B/'experiments/board_table_rectfixed_v2'
def save(n,d):(E/n).write_text(json.dumps(d,indent=2),encoding='utf-8')
def main():
    E.mkdir(exist_ok=True);assert not (E/'locked_plan.json').exists()
    manifest=B/'board_table_crops_v2/manifest.json';rows=[r for r in json.loads(manifest.read_text(encoding='utf-8'))['records'] if r['accepted']]
    weights=B/'yolo11s.pt'
    plan={'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'backbone':str(weights),'backbone_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'rect':False,'feature_preprocessing':'fixed 640 square letterbox independent of batch shapes','seed':20261027,'epochs':120,'optimizer':'Adam','lr':.01,'head':'L2-normalized frozen original YOLO embeddings -> Linear(embedding_dim,2)','checkpoint':'fixed last epoch; no validation checkpoint/threshold selection','gate':{'old_correct_min':5,'new_board_recall_min':.7,'new_table_recall_min':.8},'scope':'67 photo crops, 46 training, old8 regression + new13 holdout. Same frozen features/120 epochs/lr. Only data expansion treatment. No VLM or YOLO backbone fine-tuning. Public COCO pretraining may overlap LVIS; not independent accuracy. No adoption.'}
    save('locked_plan.json',plan)
    import torch
    from ultralytics import YOLO
    random.seed(plan['seed']);torch.manual_seed(plan['seed'])
    model=YOLO(str(weights))
    vectors=model.embed([r['image'] for r in rows],imgsz=640,rect=False,batch=8,device=0,verbose=False)
    x=torch.nn.functional.normalize(torch.stack([v.detach().float().cpu() for v in vectors]),dim=1)
    y=torch.tensor([0 if r['class']=='board' else 1 for r in rows]);train=torch.tensor([r['pilot_split']=='train' for r in rows]);val=~train
    head=torch.nn.Linear(x.shape[1],2);optimizer=torch.optim.Adam(head.parameters(),lr=.01)
    counts=torch.bincount(y[train],minlength=2).float();class_weights=counts.sum()/(2*counts)
    history=[]
    for epoch in range(120):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(head(x[train]),y[train],weight=class_weights);loss.backward();optimizer.step();history.append(float(loss.detach()))
    torch.save({'state_dict':head.state_dict(),'embedding_dim':x.shape[1],'classes':['board','table'],'plan':plan},E/'head_last.pt')
    with torch.no_grad():probs=head(x).softmax(1);pred=probs.argmax(1)
    results=[]
    for i,r in enumerate(rows):results.append({'image':r['image'],'class':r['class'],'split':r['pilot_split'],'predicted':['board','table'][int(pred[i])],'probabilities':probs[i].tolist(),'source_photo_id':r['source_photo_id']})
    wood=json.loads((B/'region_new_sources_v1/manifest.json').read_text(encoding='utf-8'))['records'];wood=[r for r in wood if r['role']=='wood_table_negative']
    embedded=model.embed([r['image'] for r in wood],imgsz=640,rect=False,batch=3,device=0,verbose=False)
    with torch.no_grad():wood_probs=head(torch.nn.functional.normalize(torch.stack([v.detach().float().cpu() for v in embedded]),dim=1)).softmax(1)
    diagnostics=[{'image':r['image'],'predicted':['board','table'][int(p.argmax())],'probabilities':p.tolist(),'scope':'Previously exposed wood video, full image vs crop domain shift. Not selection input.'} for r,p in zip(wood,wood_probs)]
    old=torch.load(B/'experiments/board_table_rectfixed_v1/head_last.pt',map_location='cpu',weights_only=True)
    baseline=torch.nn.Linear(old['embedding_dim'],2);baseline.load_state_dict(old['state_dict'])
    with torch.no_grad():baseline_prob=baseline(x).softmax(1)
    save('previous_head_predictions.json',[{'image':r['image'],'class':r['class'],'split':r['pilot_split'],'predicted':['board','table'][int(p.argmax())]} for r,p in zip(rows,baseline_prob)])
    save('results.json',{'rows':results,'wood_development':diagnostics,'loss_history':history,'adopted':False,'backbone_trained':False,'vlm_trained':False})
    save('status.json',{'stage':'completed','adopted':False});print({s:sum(r['class']==r['predicted'] for r in results if r['split']==s) for s in ['train','old_validation','new_validation']},diagnostics,flush=True)
if __name__=='__main__':main()
