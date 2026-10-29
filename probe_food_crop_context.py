"""Paired crop-context sensitivity, fixed prompt and frozen model."""
import json,hashlib,math
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_crop_context_v1'
IDS=['pepper_04_gloved_hand','pepper_04_pepper_group','pepper_04_single_pepper_piece','meat_01_hand','meat_01_small_meat','meat_02_garlic_group','pork_01_hand','pork_01_small_meat','chicken_05_small_meat','chicken_05_wing','chicken_08_hand','chicken_08_meat_group']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 E.mkdir(exist_ok=False)
 m=B/'food_presence_patches_v3/reviewed_manifest.json';byid={r['id']:r for r in json.loads(m.read_text(encoding='utf-8'))['records']};cases=[]
 for key in IDS:
  r=byid[key];assert sha(r['image'])==r['image_sha256'];im=Image.open(r['image']).convert('RGB');a,b,c,d=r['crop_xyxy'];w=c-a;h=d-b
  for margin in [0,.5]:
   box=[max(0,math.floor(a-margin*w)),max(0,math.floor(b-margin*h)),min(im.width,math.ceil(c+margin*w)),min(im.height,math.ceil(d+margin*h))]
   p=E/f'{key}_{margin}.jpg';im.crop(box).save(p,quality=95)
   cases.append(dict(id=key,margin=margin,reference=r['label'],image=r['image'],image_sha256=r['image_sha256'],source_group=r['source_group'],box=box,crop=str(p),crop_sha256=sha(p),reference_scope='Reference applies to original tight crop only; wider context can contain additional food.'))
 from food_category_review import FOOD_CATEGORY_PROMPT,CATEGORIES
 save('locked_plan.json',dict(cases=cases,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12,scope='12 selected development cases x 2 paired crops. Not independent accuracy; wide crop can include different food. No adoption.'))
 from region_vlm_runtime import LocalRegionModel
 model=LocalRegionModel();rows=[]
 for case in cases:
  im=Image.open(case['image']).convert('RGB').crop(case['box']);raw=model.predict(im,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12)
  rows.append(dict(**case,raw_response=raw,opinion=raw.strip().lower() if raw.strip().lower() in CATEGORIES else 'unknown'))
  save('partial_results.json',dict(rows=rows));print(case['id'],case['margin'],rows[-1]['opinion'],flush=True)
 save('results.json',dict(rows=rows,revision=model.revision,training_performed=False))
 for page in range(3):
  sheet=Image.new('RGB',(1200,1200),'white');draw=ImageDraw.Draw(sheet)
  for k,r in enumerate(rows[page*8:page*8+8]):
   im=Image.open(r['crop']);im.thumbnail((290,450));x=k%4*300;y=k//4*600;sheet.paste(im,(x,y));draw.text((x,y+460),r['id'],fill='black');draw.text((x,y+480),str(r['margin'])+' '+r['opinion'],fill='black')
  sheet.save(E/f'review_{page}.jpg')
if __name__=='__main__':main()
