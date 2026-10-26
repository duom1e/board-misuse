"""Assistant-reviewed coarse food-group boxes, never automatic colour labels."""
import json, hashlib, shutil
from pathlib import Path
from PIL import Image, ImageDraw

B = Path(__file__).resolve().parent
OUT = B / 'canteen_food_boxes_v1'
# Coordinates on the 264x215 review thumbnails inspected by the annotator.
# A box encloses a same-class food group, not each individual slice.
BOXES = {
 10:[(0,31,190,102)],15:[(0,31,190,102)],20:[(0,31,190,102)],
 25:[(0,31,190,102)],30:[(50,11,108,54)],
 45:[(37,24,126,90)],50:[(37,24,126,90)],55:[(37,24,126,90)],
 65:[(55,57,153,102)],70:[(45,46,155,102)],74:[(82,50,153,99)],
 93:[(16,30,174,83)],98:[(44,42,214,108)],103:[(64,49,152,124)],
 108:[(53,43,175,99)],113:[(60,52,132,96)],
 138:[(77,26,168,118)],143:[(59,28,166,118)],148:[(62,47,136,96)],
 153:[(50,36,143,115)],168:[(61,53,177,138)],173:[(72,51,177,137)],
 239:[(94,44,197,105)],244:[(53,12,216,112)],249:[(34,18,216,111)],
 254:[(95,22,217,126)],275:[(41,42,212,150)],280:[(43,39,201,151)],
 285:[(65,42,202,146)]
}

def main():
 OUT.mkdir(exist_ok=True)
 source=json.loads((B/'canteen_roi_food_v1/manifest.json').read_text(encoding='utf-8'))
 records=[]
 for row in source['records']:
  im=Image.open(row['image']).convert('RGB');w,h=im.size
  boxes=[]
  for x1,y1,x2,y2 in BOXES.get(row['seconds'],[]):
   box=[max(0,round(x1*w/264)),max(0,round(y1*h/215)),min(w,round(x2*w/264)),min(h,round(y2*h/215))]
   assert 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h
   boxes.append(box)
  assert bool(boxes)==(row['class']!='empty'),row
  name=Path(row['image']).stem
  ip=OUT/'images'/row['split']/(name+'.png');ip.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['image'],ip)
  lp=OUT/'labels'/row['split']/(name+'.txt');lp.parent.mkdir(parents=True,exist_ok=True)
  cls=0 if row['class']=='meat' else 1
  lines=[f'{cls} {(a+c)/(2*w):.8f} {(b+d)/(2*h):.8f} {(c-a)/w:.8f} {(d-b)/h:.8f}' for a,b,c,d in boxes]
  lp.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
  records.append({**row,'image':str(ip),'label':str(lp),'boxes_xyxy':boxes,'annotation':'assistant_manual_coarse_visible_food_group','expert_review':False,'label_sha256':hashlib.sha256(lp.read_bytes()).hexdigest()})
 for category in ['meat','vegetable','empty']:
  rows=[r for r in records if r['class']==category]
  sheet=Image.new('RGB',(4*417,((len(rows)+3)//4)*369),'white')
  for i,r in enumerate(rows):
   im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im)
   for box in r['boxes_xyxy']:d.rectangle(box,outline='lime',width=3)
   x=i%4*417;y=i//4*369;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+344),f'{r["split"]} {category} {r["seconds"]}s',fill='black')
  sheet.save(OUT/(category+'_boxes_review.jpg'))
 (OUT/'manifest.json').write_text(json.dumps({'records':records,'review_status':'pending_overlay_review','scope':'Same-video food-group localization development data, NOT independent test; class empty means no obvious bulk food on board only. Off-board food is outside annotation scope: not safe as full-frame food negatives.','historical_test_already_consumed':True,'source_license':'CC BY-NC 4.0 local research only','annotation_policy':'Coarse box of visible same-class food group on board; may span inter-slice gaps and occluding knife/hand; not instance segmentation or every-food-in-image GT.'},ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'data.yaml').write_text(f'path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: meat\n  1: vegetable\n',encoding='utf-8')
 print(json.dumps({'images':len(records),'positive_groups':sum(len(r['boxes_xyxy']) for r in records),'negative_images':sum(not r['boxes_xyxy'] for r in records)}))

if __name__=='__main__':main()
