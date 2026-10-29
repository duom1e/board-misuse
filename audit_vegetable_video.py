import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'vegetable_video_full_v1'
def main():
 rows=json.loads((E/'recognition/results.json').read_text(encoding='utf-8'))['rows'];sheet=Image.new('RGB',(1280,900));d=ImageDraw.Draw(sheet);crops=[]
 for i,r in enumerate(rows):
  p=Path(r['image']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['image_sha256'];im=Image.open(p).convert('RGB');im.thumbnail((320,210));x=i%4*320;y=i//4*300;sheet.paste(im,(x,y));d.text((x,y+215),f"{r['frame_seconds']:.3f}s {r['board_state']}",fill='white')
  for j,q in enumerate(r.get('food_regions',[])):
   if 'category_crop_xyxy' in q:
    crop=Image.open(p).crop(q['category_crop_xyxy']);name=f'crop_{i:02}_{j:02}.jpg';crop.save(E/name);crops.append(dict(seconds=r['frame_seconds'],image=name,category=q['food_category'],location=q['location'],category_raw=q.get('category_raw_response')))
 sheet.save(E/'source_review.jpg');s=Image.new('RGB',(1280,max(1,(len(crops)+3)//4)*300));d=ImageDraw.Draw(s)
 for i,r in enumerate(crops):
  im=Image.open(E/r['image']);im.thumbnail((310,240));x=i%4*320;y=i//4*300;s.paste(im,(x,y));d.text((x,y+245),f"{r['seconds']:.3f}s {r['category']}",fill='white')
 s.save(E/'category_review.jpg');audit=dict(samples=len(rows),valid=sum(r['output_validation']['valid'] for r in rows),board_candidates=sum(r['board_state']=='candidate_present' for r in rows),category_crops=crops,category_counts={k:sum(r['category']==k for r in crops) for k in ['meat','vegetable','mixed','other','unknown']},training_performed=False,independent_accuracy=False)
 (E/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in audit.items() if k!='category_crops'})
if __name__=='__main__':main()
