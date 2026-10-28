import os,json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'round_board_sources_v1'
CASES=[('BV1hK411u73L',1,[0,0,1278,720]),('BV1hK411u73L',4,[0,45,1278,720]),('BV1hK411u73L',8,[0,0,1278,720]),('BV15J411D718',1,[310,250,1000,720]),('BV15J411D718',4,[352,257,970,720]),('BV15J411D718',8,[217,216,1102,720])]
def main():
 from ultralytics import YOLO
 assert not (E/'locked_boxes.json').exists(),'Already started; do not overwrite'
 records=[]
 for id,i,box in CASES:
  p=E/f'{id}_{i:02}.jpg';records.append(dict(id=id,index=i,box=box,image=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),annotation='assistant nonexpert before predictions',split='diagnostic_only_not_train'))
 (E/'locked_boxes.json').write_text(json.dumps(dict(records=records,conf=.25,iou_match=.5,imgsz=640,rect=False,scope='Six selected frames, two videos; three cases per source correlated. Same author/scene independence and pretraining overlap not established.'),indent=2),encoding='utf-8')
 weights=dict(baseline=json.loads((B/'active_models.json').read_text(encoding='utf-8'))['board']['weights'],ordinary=str(B/'kitchen_cutting_trial_v1/train/weights/last.pt'),conservative=str(B/'kitchen_cutting_trial_conservative_v1/train/weights/last.pt'));rows=[]
 for name,w in weights.items():
  m=YOLO(w);s=Image.new('RGB',(1280,1980))
  for i,r in enumerate(records):
   im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im);a,b,c,k=r['box'];d.rectangle(r['box'],outline='lime',width=2);out=[]
   for q in m.predict(r['image'],conf=.25,imgsz=640,rect=False,verbose=False)[0].boxes:
    xy=q.xyxy[0].tolist();x,y,z,v=xy;inter=max(0,min(c,z)-max(a,x))*max(0,min(k,v)-max(b,y));iou=inter/((c-a)*(k-b)+(z-x)*(v-y)-inter);out.append(dict(box=xy,conf=float(q.conf[0]),iou=iou));d.rectangle(xy,outline='orange',width=3)
   rows.append(dict(model=name,id=r['id'],index=r['index'],predictions=out,matched=any(q['iou']>=.5 for q in out)));im.thumbnail((640,640));x=i%2*640;y=i//2*660;s.paste(im,(x,y));ImageDraw.Draw(s).text((x,y+642),f"{name} {r['id']} {r['index']} match={rows[-1]['matched']}",fill='white')
  s.save(E/f'{name}_review.jpg')
 summary={n:sum(r['matched'] for r in rows if r['model']==n) for n in weights}
 (E/'results.json').write_text(json.dumps(dict(rows=rows,matches=summary,training_performed=False,adopted=False),indent=2),encoding='utf-8');print(summary)
if __name__=='__main__':main()
