import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
from region_vlm_runtime import LocalRegionModel
from evaluate_qwen4b_regions import PROMPT
from region_response_validation import validate_region_response
B=Path(__file__).resolve().parent;E=B/'crosssource_vlm_v1'
def save(n,o):(E/n).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 E.mkdir(exist_ok=True);assert not (E/'locked_plan.json').exists(),'Do not restart'
 pos=json.loads((B/'kitchen_positive_sources_v1/locked_boxes.json').read_text(encoding='utf-8'))['records'];neg=json.loads((B/'kitchen_multisource_v1/negative_plan.json').read_text(encoding='utf-8'))['frames'];cases=[]
 for r in pos:cases.append(dict(image=r['image'],sha256=r['sha256'],expected=r['box']))
 for r in neg:cases.append(dict(image=r['image'],sha256=r['sha256'],expected=None))
 save('locked_plan.json',dict(cases=cases,prompt=PROMPT,max_new_tokens=300,threshold_iou=.5,scope='Previously selected manual diagnostic cases, no tuning or training; source correlated. Unknown schema never negative.'))
 m=LocalRegionModel();rows=[]
 for i,r in enumerate(cases):
  p=Path(r['image']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];im=Image.open(p).convert('RGB');raw=m.predict(im,prompt=PROMPT,max_new_tokens=300);v=validate_region_response(raw);pred=[];w,h=im.size;d=ImageDraw.Draw(im)
  if r['expected']:d.rectangle(r['expected'],outline='lime',width=2)
  if v['valid']:
   for q in v['normalized_boxes']['boards']:
    x,y,z,k=[q[0]*w/1000,q[1]*h/1000,q[2]*w/1000,q[3]*h/1000];iou=None
    if r['expected']:
     a,b,c,f=r['expected'];inter=max(0,min(c,z)-max(a,x))*max(0,min(f,k)-max(b,y));iou=inter/((c-a)*(f-b)+(z-x)*(k-y)-inter)
    pred.append(dict(box=[x,y,z,k],iou=iou));d.rectangle([x,y,z,k],outline='orange',width=3)
  im.save(E/f'{i:02}_review.jpg');row=dict(**r,raw_response=raw,validation=v,boards=pred,matched=any((x['iou'] or 0)>=.5 for x in pred));rows.append(row);save('partial_results.json',dict(rows=rows));print(i,v['valid'],row['matched'],len(pred),flush=True)
 save('results.json',dict(rows=rows,training_performed=False,adopted=False))
if __name__=='__main__':main()
