import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_group_boxes_v3'
SPECS=[(4,[190,360,790,590],[[215,400,335,560],[365,400,625,480],[610,375,780,490],[585,490,745,585]]),(9,[290,365,780,590],[[315,410,565,495],[620,375,760,475],[570,445,760,585]])]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);data=json.loads((B/'food_group_boxes_v2/manifest.json').read_text(encoding='utf-8'));sources=json.loads((B/'yellow_chicken_sources_v1/sources.json').read_text(encoding='utf-8'));src=next(s for s in sources if s['id']=='BV1Go4y1o7ym')
 for i,roi,boxes in SPECS:
  f=src['frames'][i];im=Image.open(f['image']).convert('RGB').crop(roi);p=E/'images'/f'yellow_{i}.jpg';p.parent.mkdir(exist_ok=True);im.save(p,quality=95);local=[[a-roi[0],b-roi[1],c-roi[0],d-roi[1]] for a,b,c,d in boxes]
  label=E/'labels'/f'yellow_{i}.txt';label.parent.mkdir(exist_ok=True);label.write_text(''.join(f'0 {(a+c)/2/im.width:.8f} {(b+d)/2/im.height:.8f} {(c-a)/im.width:.8f} {(d-b)/im.height:.8f}\n' for a,b,c,d in local),encoding='utf-8')
  data['records'].append(dict(id=f'yellow_{i}',image=str(p),sha256=sha(p),label=str(label),label_sha256=sha(label),source_group=src['id'],source_frame=f['image'],source_frame_sha256=sha(f['image']),video_sha256=src['sha256'],roi=roi,boxes_xyxy=local,category='meat',split='training_candidate',review_status='pending_overlay'))
  for box in local:ImageDraw.Draw(im).rectangle(box,outline='lime',width=3)
  im.save(E/f'yellow_{i}_overlay.jpg')
 (E/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
