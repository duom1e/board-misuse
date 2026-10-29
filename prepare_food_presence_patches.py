"""Visually specified crop-category supervision; no pseudo object boxes."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_presence_patches_v1'
# Original 1280/1270 x 720 coordinates; no context margin added.
SPECS=[
('pepper',4,'gloved_hand','background',[790,0,1030,120]),
('pepper',4,'knife_face','background',[820,310,915,400]),
('pepper',4,'board','background',[70,300,330,500]),
('pepper',4,'pepper_group','vegetable',[440,185,755,435]),
('pepper',4,'single_pepper_piece','vegetable',[710,475,835,570]),
('pepper',4,'whole_peppers','vegetable',[1050,140,1260,300]),
('pepper',6,'gloved_left','background',[55,160,320,425]),
('pepper',6,'gloved_right','background',[1120,230,1260,475]),
('pepper',6,'knife_face','background',[520,345,900,430]),
('pepper',6,'pepper_group','vegetable',[460,30,1000,255]),
('meat',1,'hand','background',[700,165,840,255]),
('meat',1,'knife_face','background',[685,325,745,370]),
('meat',1,'board','background',[420,430,710,600]),
('meat',1,'small_meat','meat',[480,215,580,285]),
('meat',1,'meat_group','meat',[525,260,675,400]),
('meat',1,'bowl_vegetable','vegetable',[125,235,320,425]),
('meat',2,'knife_face','background',[540,60,640,155]),
('meat',2,'board','background',[490,400,800,550]),
('meat',2,'garlic_group','vegetable',[525,215,760,310]),
('meat',2,'bowl_meat','meat',[150,505,315,625]),
]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);rows=[]
 maps={}
 for group,folder in [('pepper','hand_food_sources_v1'),('meat','minced_food_sources_v1')]:
  maps[group]=json.loads((B/folder/'sources.json').read_text(encoding='utf-8'))[0]
 for group,index,name,label,box in SPECS:
  source=maps[group];f=source['frames'][index];assert sha(f['image'])==f['sha256']
  im=Image.open(f['image']).convert('RGB');assert 0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height
  key=f'{group}_{index:02d}_{name}';p=E/f'{key}.jpg';im.crop(box).save(p,quality=95)
  rows.append(dict(id=key,source_group=source['id'],source_video=source['video'],source_video_sha256=source['sha256'],source_url=source['url'],image=f['image'],image_sha256=f['sha256'],seconds=f['seconds'],frame_index=f['frame_index'],crop_xyxy=box,crop=str(p),crop_sha256=sha(p),label=label,review_status='pending_crop_visual_check',training_eligible=False,split='consumed_development',physical_contact_verified=False))
 for start in range(0,len(rows),10):
  sheet=Image.new('RGB',(1250,600),'white');d=ImageDraw.Draw(sheet)
  for k,r in enumerate(rows[start:start+10]):
   im=Image.open(r['crop']);im.thumbnail((240,230));x=k%5*250;y=k//5*300;sheet.paste(im,(x,y));d.text((x+3,y+235),r['id'],fill='black');d.text((x+3,y+255),r['label'],fill='black')
  sheet.save(E/f'review_{start//10}.jpg')
 (E/'manifest.json').write_text(json.dumps(dict(records=rows,scope='Exact crop classification only; background means no clear meat/plant ingredient visible in this selected crop, not a negative label for its full frame. Nonexpert visual annotations; sources previously consumed. No random adjacent-frame split. Media reuse license unverified; local work only.',training_performed=False),ensure_ascii=False,indent=2),encoding='utf-8')
 print(len(rows),'patches created')
if __name__=='__main__':main()
