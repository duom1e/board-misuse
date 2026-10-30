import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_group_boxes_v4'
SPECS=[(2,[[130,0,465,310],[345,195,1015,600]]),(3,[[90,15,265,220],[45,230,300,390],[25,380,305,520],[30,525,315,680],[475,60,710,320],[440,300,820,500],[450,490,855,680],[905,65,1205,275],[965,305,1215,705]])]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);data=json.loads((B/'food_group_boxes_v3/manifest.json').read_text(encoding='utf-8'));sources=json.loads((B/'yellow_chicken_sources_v1/sources.json').read_text(encoding='utf-8'));src=next(s for s in sources if s['id']=='BV1PW411F7mD')
 for i,boxes in SPECS:
  f=src['frames'][i];im=Image.open(f['image']).convert('RGB');assert sha(f['image'])==f['sha256'];p=E/'images'/f'bamboo_{i}.jpg';p.parent.mkdir(exist_ok=True);im.save(p,quality=95)
  label=E/'labels'/f'bamboo_{i}.txt';label.parent.mkdir(exist_ok=True);label.write_text(''.join(f'0 {(a+c)/2/im.width:.8f} {(b+d)/2/im.height:.8f} {(c-a)/im.width:.8f} {(d-b)/im.height:.8f}\n' for a,b,c,d in boxes),encoding='utf-8')
  data['records'].append(dict(id=f'bamboo_{i}',image=str(p),sha256=sha(p),label=str(label),label_sha256=sha(label),source_group=src['id'],source_frame=f['image'],source_frame_sha256=f['sha256'],video_sha256=src['sha256'],roi=[0,0,im.width,im.height],boxes_xyxy=boxes,category='meat',split='training_candidate',review_status='pending_overlay'))
  for box in boxes:ImageDraw.Draw(im).rectangle(box,outline='lime',width=3)
  im.save(E/f'bamboo_{i}_overlay.jpg')
 data['scope']='Prior group annotations plus2 bamboo-board frames11 coarse food groups, nonexpert. Original held source unchanged; all sources consumed development.'
 (E/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
