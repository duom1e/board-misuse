import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_group_boxes_v2'
SPECS=[(1,[200,0,1278,520],[[225,0,1278,480]]),(4,[260,0,1010,650],[[295,0,790,650]]),(6,[160,0,1278,620],[[190,115,535,595],[625,0,1005,610],[950,0,1278,620]]),(8,[175,0,1120,720],[[190,235,490,365],[400,0,1110,720]])]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);data=json.loads((B/'food_group_boxes_v1/manifest.json').read_text(encoding='utf-8'));src=json.loads((B/'round_board_sources_v1/sources.json').read_text(encoding='utf-8'))[0];added=[]
 for i,roi,boxes in SPECS:
  if i==6:continue  # Overlay review found unlabeled bowl food at the ROI edge.
  f=src['frames'][i];assert sha(f['image'])==f['sha256'];im=Image.open(f['image']).convert('RGB').crop(roi);key=f'{src["id"]}_{i:02d}';p=E/'images'/f'{key}.jpg';p.parent.mkdir(exist_ok=True);im.save(p,quality=95)
  local=[[a-roi[0],b-roi[1],c-roi[0],d-roi[1]] for a,b,c,d in boxes];assert all(0<=a<c<=im.width and 0<=b<d<=im.height for a,b,c,d in local)
  label=E/'labels'/f'{key}.txt';label.parent.mkdir(exist_ok=True);label.write_text(''.join(f'0 {(a+c)/2/im.width:.8f} {(b+d)/2/im.height:.8f} {(c-a)/im.width:.8f} {(d-b)/im.height:.8f}\n' for a,b,c,d in local),encoding='utf-8')
  for box in local:ImageDraw.Draw(im).rectangle(box,outline='lime',width=3)
  im.save(E/f'{key}_overlay.jpg');added.append(dict(id=key,image=str(p),sha256=sha(p),label=str(label),label_sha256=sha(label),source_group=src['id'],source_frame=f['image'],source_frame_sha256=f['sha256'],video_sha256=src['sha256'],roi=roi,boxes_xyxy=local,category='meat',split='training_candidate',review_status='pending_overlay'))
 sheet=Image.new('RGB',(1200,1000),'white')
 for i,r in enumerate(added):
  im=Image.open(E/f'{r["id"]}_overlay.jpg');im.thumbnail((590,460));x=i%2*600;y=i//2*500;sheet.paste(im,(x,y))
 sheet.save(E/'review.jpg');data['records']+=added;data['scope']+=' Added four consumed-source chicken ROIs, 7 group boxes. Original chicken held source unchanged.'
 (E/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
