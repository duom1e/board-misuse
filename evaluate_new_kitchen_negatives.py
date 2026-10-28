"""New source negative-frame diagnostic; no tuning or automatic adoption."""
import os,json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'kitchen_multisource_v1'
def main():
    from ultralytics import YOLO
    manifest=E/'negative_plan.json';assert not manifest.exists(),'Already started, inspect existing results'
    source=json.loads((E/'sources.json').read_text(encoding='utf-8'))[0]
    # Chosen after contact-sheet viewing, before any predictions on this source.
    frames=[source['frames'][i] for i in (3,4,5,7)]
    manifest.write_text(json.dumps(dict(frames=frames,expected='no clearly visible board',review='assistant nonexpert before model outputs',scope='Four correlated selected frames from new project source, not full source absence',conf=.25,imgsz=640,rect=False),indent=2),encoding='utf-8')
    weights=dict(baseline=json.loads((B/'active_models.json').read_text(encoding='utf-8'))['board']['weights'],ordinary=str(B/'kitchen_cutting_trial_v1/train/weights/last.pt'),conservative=str(B/'kitchen_cutting_trial_conservative_v1/train/weights/last.pt'))
    rows=[]
    for name,w in weights.items():
        model=YOLO(w);sheet=Image.new('RGB',(1280,760))
        for i,f in enumerate(frames):
            p=Path(f['image']);assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
            pred=model.predict(str(p),imgsz=640,rect=False,conf=.25,verbose=False)[0]
            boxes=[dict(box=q.xyxy[0].tolist(),confidence=float(q.conf[0])) for q in pred.boxes];rows.append(dict(model=name,image=str(p),index=f['index'],seconds=f['seconds'],boxes=boxes))
            im=Image.open(p).convert('RGB');d=ImageDraw.Draw(im)
            for q in boxes:d.rectangle(q['box'],outline='orange',width=3)
            im.thumbnail((640,350));x=i%2*640;y=i//2*380;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x,y+355),f"{f['index']} {name} {len(boxes)} candidates",fill='white')
        sheet.save(E/f'negative_{name}.jpg')
    (E/'negative_results.json').write_text(json.dumps(dict(rows=rows,training_performed=False,adopted=False),indent=2),encoding='utf-8');print({n:sum(bool(r['boxes']) for r in rows if r['model']==n) for n in weights})
if __name__=='__main__':main()
