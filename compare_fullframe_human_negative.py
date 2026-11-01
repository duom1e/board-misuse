import os,json,hashlib,html
from pathlib import Path
from PIL import Image,ImageDraw
os.environ['YOLO_OFFLINE']='True'
from ultralytics import YOLO
B=Path(__file__).resolve().parent;E=B/'fullframe_human_negative_comparison_v1'
def main():
 E.mkdir(exist_ok=False);cases=json.loads((B/'food_batch36_v2/locked_plan.json').read_text())['cases'];out=[];models={'small':B/'food_fullmask_multiscale45_v1/training/weights/last.pt','expanded':B/'food_human_negative10_v1/training/weights/last.pt'}
 for name,w in models.items():
  model=YOLO(str(w));model.model.cpu();rows=[]
  for i,r in enumerate(cases):
   assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==r['sha256']
   p=model.predict(r['image'],conf=.25,imgsz=640,rect=False,device='cpu',verbose=False)[0];ps=[dict(box=q.xyxy[0].tolist(),category=model.names[int(q.cls[0])],confidence=float(q.conf[0])) for q in p.boxes];im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im)
   for q in ps:d.rectangle(q['box'],outline='red' if q['category']=='meat' else 'lime',width=3);d.text((q['box'][0],q['box'][1]),q['category'],fill='red')
   im.thumbnail((640,400));im.save(E/f'{name}_{i}.jpg');rows.append(dict(index=i,source=r['source'],seconds=r['seconds'],predictions=ps))
  out.append(dict(model=name,weight_sha256=hashlib.sha256(w.read_bytes()).hexdigest(),rows=rows))
 (E/'results.json').write_text(json.dumps(out,indent=2))
 parts=['<!doctype html><meta charset="utf-8"><title>全画面对照</title><style>body{font:16px sans-serif;max-width:1400px;margin:auto}section{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%}</style><h1>第28次基线/人体负例候选：完整画面候选框</h1><p>36张历史画面，部分来源已用于训练；没有完整框真值，不计算准确率。红色肉、绿色植物食材。无框不等于无食材，不判断案板接触或错用。</p>']
 for i,r in enumerate(cases):parts.append(f'<h2>{i} {html.escape(r["source"])} {r["seconds"]:.1f}s</h2><section><div>第28次基线<img src="small_{i}.jpg"></div><div>人体负例候选<img src="expanded_{i}.jpg"></div></section>')
 (E/'index.html').write_text('\n'.join(parts),encoding='utf-8')
 for page in range(3):
  sheet=Image.new('RGB',(1280,1800))
  for j,i in enumerate(range(page*12,(page+1)*12)):
   for col,name in enumerate(models):
    im=Image.open(E/f'{name}_{i}.jpg');im.thumbnail((310,270));x=(j%2)*640+col*320;y=(j//2)*300;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x,y+275),f'{i} {name}',fill='white')
  sheet.save(E/f'page_{page}.jpg')
 print([(x['model'],sum(bool(r['predictions']) for r in x['rows']),sum(len(r['predictions']) for r in x['rows'])) for x in out])
if __name__=='__main__':main()
