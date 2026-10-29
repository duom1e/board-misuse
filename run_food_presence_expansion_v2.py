"""Fixed source expansion comparison; old fold settings remain unchanged."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_presence_patches_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False)
 old=json.loads((B/'food_presence_patches_v2/reviewed_manifest.json').read_text(encoding='utf-8'))
 s=json.loads((B/'pepper_presence_sources_v2/sources.json').read_text(encoding='utf-8'))[0]
 specs=[(3,'board','background',[80,140,310,300]),(3,'whole_pepper','vegetable',[560,210,970,525]),(3,'counter','background',[30,450,150,550]),(4,'garlic','vegetable',[520,240,825,530]),(4,'pepper','vegetable',[895,370,1035,530]),(4,'counter','background',[25,190,145,360])]
 added=[]
 for i,name,label,box in specs:
  f=s['frames'][i];assert sha(f['image'])==f['sha256'];p=E/f'newpepper_{i}_{name}.jpg'
  Image.open(f['image']).crop(box).save(p,quality=95)
  added.append(dict(id=p.stem,source_group=s['id'],source_video=s['video'],source_video_sha256=s['sha256'],source_url=s['url'],image=f['image'],image_sha256=f['sha256'],seconds=f['seconds'],frame_index=f['frame_index'],crop_xyxy=box,crop=str(p),crop_sha256=sha(p),label=label,training_eligible=True,detection_training_eligible=False,training_scope='local_experimental_crop_classification_only',review_status='assistant_visual_nonexpert',physical_contact_verified=False))
 sheet=Image.new('RGB',(1200,600),'white');d=ImageDraw.Draw(sheet)
 for i,r in enumerate(added):
  im=Image.open(r['crop']);im.thumbnail((380,230));x=i%3*400;y=i//3*300;sheet.paste(im,(x,y));d.text((x,y+240),r['id']+' '+r['label'],fill='black')
 sheet.save(E/'review.jpg')
 (E/'reviewed_manifest.json').write_text(json.dumps(dict(records=old['records']+added,scope='42 crop development references across 5 consumed videos; nonexpert. No independent accuracy.'),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
