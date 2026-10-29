"""Nonexpert visible food-group boxes, explicitly bounded ROI scope."""
import json,hashlib,shutil
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_group_boxes_v1'
# ROI excludes subtitles and off-board containers; nearby food fragments grouped.
SPECS=[('meat_presence_sources_v1','BV14W41197Rd',1,[250,250,1280,720],'meat',[[285,300,620,490],[620,260,1130,485],[775,545,1270,720]]),('meat_presence_sources_v1','BV14W41197Rd',3,[340,160,1040,410],'meat',[[370,175,605,405],[595,180,1010,385]]),('chicken_presence_sources_v1','BV1264y1F7a6',5,[490,20,960,720],'meat',[[590,35,950,230],[505,285,675,375],[590,370,845,505],[510,490,835,710]]),('chicken_presence_sources_v1','BV1264y1F7a6',8,[510,235,970,720],'meat',[[735,250,910,365],[525,395,950,715]]),('hand_food_sources_v1','BV14K4y1x7LF',4,[420,120,1270,580],'vegetable',[[435,180,770,440],[710,480,835,565],[1025,125,1270,315]]),('hand_food_sources_v1','BV14K4y1x7LF',6,[380,0,1090,300],'vegetable',[[385,0,1090,300]])]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);rows=[]
 for folder,source,i,roi,cat,boxes in SPECS:
  src=json.loads((B/folder/'sources.json').read_text(encoding='utf-8'))[0];f=src['frames'][i];assert sha(f['image'])==f['sha256']
  im=Image.open(f['image']).convert('RGB').crop(roi);key=f'{source}_{i:02d}';path=E/'images'/f'{key}.jpg';path.parent.mkdir(exist_ok=True);im.save(path,quality=95)
  local=[[a-roi[0],b-roi[1],c-roi[0],d-roi[1]] for a,b,c,d in boxes]
  assert all(0<=a<c<=im.width and 0<=b<d<=im.height for a,b,c,d in local)
  label=E/'labels'/f'{key}.txt';label.parent.mkdir(exist_ok=True);label.write_text(''.join(f'{int(cat=="vegetable")} {(a+c)/2/im.width:.8f} {(b+d)/2/im.height:.8f} {(c-a)/im.width:.8f} {(d-b)/im.height:.8f}\n' for a,b,c,d in local),encoding='utf-8')
  dr=ImageDraw.Draw(im)
  for box in local:dr.rectangle(box,outline='lime',width=3)
  im.save(E/f'{key}_overlay.jpg')
  rows.append(dict(id=key,image=str(path),sha256=sha(path),label=str(label),label_sha256=sha(label),source_group=source,source_frame=f['image'],source_frame_sha256=f['sha256'],video_sha256=src['sha256'],roi=roi,boxes_xyxy=local,category=cat,split='source_held_development' if source=='BV1264y1F7a6' else 'training_candidate',review_status='pending_overlay'))
 (E/'manifest.json').write_text(json.dumps(dict(records=rows,scope='Coarse visible food-group boxes inside manually bounded ROIs, not full-frame exhaustive instance annotations; tiny seeds/specks not individual targets. Nonexpert, historical consumed videos; no blind test. Chicken source reserved from this new training proposal.',training_performed=False),ensure_ascii=False,indent=2),encoding='utf-8')
 sheet=Image.new('RGB',(1500,1000),'white')
 for i,r in enumerate(rows):
  im=Image.open(E/f'{r["id"]}_overlay.jpg');im.thumbnail((490,450));x=i%3*500;y=i//3*500;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x,y+460),r['id'],fill='black')
 sheet.save(E/'review.jpg');print(len(rows),sum(len(r['boxes_xyxy']) for r in rows))
if __name__=='__main__':main()
