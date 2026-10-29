"""Partial coarse-food GT audit, separating localization from category opinions."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_localization_audit_v1'
def iou(a,b):
 x,y,z,k=a;v,w,f,h=b;s=max(0,min(z,f)-max(x,v))*max(0,min(k,h)-max(y,w));return s/((z-x)*(k-y)+(f-v)*(h-w)-s)
def main():
 E.mkdir(exist_ok=True);assert not (E/'manifest.json').exists(),'Do not overwrite reviewed labels'
 meat=json.loads((B/'round_food_chain_v1/results.json').read_text(encoding='utf-8'))['rows'];veg=json.loads((B/'vegetable_video_full_v1/recognition/results.json').read_text(encoding='utf-8'))['rows']
 selected=[(meat[4],[( [390,320,825,665],'meat'),([955,330,1020,375],'meat'),([1125,525,1280,720],'meat')]),(meat[5],[([275,330,1130,720],'meat')]),(veg[3],[([230,24,350,285],'vegetable'),([365,114,425,202],'vegetable')]),(veg[6],[([60,35,545,325],'vegetable')])]
 records=[];summary=[];sheet=Image.new('RGB',(1280,850))
 for i,(row,annotations) in enumerate(selected):
  p=Path(row['image']);im=Image.open(p).convert('RGB');w,h=im.size;d=ImageDraw.Draw(im);pred=row.get('food_regions',row.get('regions',[]));used=set()
  for j,(box,category) in enumerate(annotations):
   a,b,c,k=box;assert 0<=a<c<=w and 0<=b<k<=h;d.rectangle(box,outline='lime',width=3)
   score,index=max(((iou(box,q['box']),n) for n,q in enumerate(pred) if n not in used),default=(0,None));matched=score>=.5
   if matched:used.add(index)
   q=pred[index] if matched else {};summary.append(dict(image=i,gt_index=j,expected=category,localization_matched=matched,iou=score if matched else None,category=q.get('food_category'),location=q.get('location'),category_inference_performed=q.get('category_inference_performed',False)))
  for q in pred:d.rectangle(q['box'],outline='orange',width=2)
  records.append(dict(image=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),annotations=[dict(box=box,category=cat) for box,cat in annotations],scope='Selected coarse food groups, not exhaustive instances. Unmatched predictions not scored as FP. Annotation after previous model exposure, nonexpert.',training_eligible=False))
  im.thumbnail((640,370));x=i%2*640;y=i//2*425;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x,y+375),str(i),fill='white')
 sheet.save(E/'overlay.jpg');(E/'manifest.json').write_text(json.dumps(dict(records=records,classes=['meat','vegetable'],training_performed=False),indent=2),encoding='utf-8');(E/'matches.json').write_text(json.dumps(dict(rows=summary,matched=sum(x['localization_matched'] for x in summary),targets=len(summary),threshold=.5,whole_image_false_positive_metric_available=False),indent=2),encoding='utf-8');print(summary)
if __name__=='__main__':main()
