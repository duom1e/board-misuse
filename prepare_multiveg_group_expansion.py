import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_group_boxes_v6'
SPECS=[(1,[[165,80,390,300]]),(3,[[235,130,455,290],[205,30,320,105],[205,140,245,185]]),(5,[[85,105,160,235],[108,100,180,260],[145,100,220,250],[185,85,265,290],[250,70,320,290],[305,70,415,335],[395,65,485,335],[435,40,535,335]]),(7,[[0,0,300,290],[310,80,520,245]])]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);data=json.loads((B/'food_group_boxes_v5/manifest.json').read_text(encoding='utf-8'));sources=json.loads((B/'kitchen_positive_sources_v1/sources.json').read_text(encoding='utf-8'));src=next(s for s in sources if s['id']=='BV15gZzYXE4N')
 for i,boxes in SPECS:
  f=src['frames'][i];im=Image.open(f['image']).convert('RGB');assert sha(f['image'])==f['sha256'];p=E/'images'/f'multiveg_{i}.jpg';p.parent.mkdir(exist_ok=True);im.save(p,quality=95)
  label=E/'labels'/f'multiveg_{i}.txt';label.parent.mkdir(exist_ok=True);label.write_text(''.join(f'1 {(a+c)/2/im.width:.8f} {(b+d)/2/im.height:.8f} {(c-a)/im.width:.8f} {(d-b)/im.height:.8f}\n' for a,b,c,d in boxes),encoding='utf-8')
  data['records'].append(dict(id=f'multiveg_{i}',image=str(p),sha256=sha(p),label=str(label),label_sha256=sha(label),source_group=src['id'],source_frame=f['image'],source_frame_sha256=f['sha256'],video_sha256=src['sha256'],roi=[0,0,im.width,im.height],boxes_xyxy=boxes,category='vegetable',split='training_candidate',review_status='pending_overlay'))
  for box in boxes:ImageDraw.Draw(im).rectangle(box,outline='lime',width=3)
  im.save(E/f'multiveg_{i}_overlay.jpg')
 data['scope']='Prior annotations plus4 multivegetable frames14 visible food-group boxes, nonexpert. Original held source unchanged; all sources consumed development.'
 (E/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
