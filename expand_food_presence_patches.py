"""Add paired meat/background crops from two other consumed video sources."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_presence_patches_v2'
SPECS=[('pork',1,'hand','background',[860,80,1040,255]),('pork',1,'knife','background',[600,60,710,180]),('pork',1,'board','background',[60,440,265,610]),('pork',1,'meat_slices','meat',[285,300,620,490]),('pork',1,'small_meat','meat',[1090,600,1250,720]),('pork',3,'hand','background',[760,35,970,175]),('pork',3,'knife','background',[50,150,250,360]),('pork',3,'meat_strips','meat',[375,180,595,400]),('chicken',5,'knife','background',[160,495,300,590]),('chicken',5,'board','background',[330,120,520,240]),('chicken',5,'small_meat','meat',[500,285,675,375]),('chicken',5,'wing','meat',[590,40,955,235]),('chicken',8,'hand','background',[80,235,330,435]),('chicken',8,'board','background',[1000,290,1150,445]),('chicken',8,'small_meat','meat',[735,255,910,365]),('chicken',8,'meat_group','meat',[530,400,945,710])]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False)
 old=json.loads((B/'food_presence_patches_v1/reviewed_manifest.json').read_text(encoding='utf-8'))
 rows=old['records']
 for r in rows:r['image']=str((B/r['image']).resolve())
 maps={g:json.loads((B/f/'sources.json').read_text(encoding='utf-8'))[0] for g,f in [('pork','meat_presence_sources_v1'),('chicken','chicken_presence_sources_v1')]}
 added=[]
 for g,i,name,label,box in SPECS:
  s=maps[g];f=s['frames'][i];assert sha(f['image'])==f['sha256'];im=Image.open(f['image']).convert('RGB')
  assert 0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height
  key=f'{g}_{i:02d}_{name}';p=E/f'{key}.jpg';im.crop(box).save(p,quality=95)
  added.append(dict(id=key,source_group=s['id'],source_video=s['video'],source_video_sha256=s['sha256'],source_url=s['url'],image=f['image'],image_sha256=f['sha256'],seconds=f['seconds'],frame_index=f['frame_index'],crop_xyxy=box,crop=str(p),crop_sha256=sha(p),label=label,review_status='pending_crop_visual_check',training_eligible=False,training_scope='local_experimental_crop_classification_only',detection_training_eligible=False,physical_contact_verified=False,split='consumed_development'))
 for page in range(2):
  sheet=Image.new('RGB',(1200,650),'white');d=ImageDraw.Draw(sheet)
  for k,r in enumerate(added[page*8:page*8+8]):
   im=Image.open(r['crop']);im.thumbnail((290,250));x=k%4*300;y=k//4*325;sheet.paste(im,(x,y));d.text((x,y+255),r['id']+' '+r['label'],fill='black')
  sheet.save(E/f'added_review_{page}.jpg')
 (E/'manifest.json').write_text(json.dumps(dict(records=rows+added,scope='36 exact crop category references from 4 previously consumed videos; nonexpert, no full-frame boxes. No VLM fine-tuning. Pending added-crop review.'),ensure_ascii=False,indent=2),encoding='utf-8')
 print(len(rows+added))
if __name__=='__main__':main()
