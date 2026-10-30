"""Lock nonexpert bounded food-group references before detector comparison."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'kitchen_source_validation_v1'
SPECS=[(2,'meat',[280,260,690,660],[315,295,655,650]),(3,'meat',[440,0,1000,520],[485,0,960,490]),(4,'vegetable',[280,120,800,610],[310,170,750,590]),(5,'meat',[370,260,755,620],[405,280,705,600])]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);s=next(r for r in json.loads((B/'yellow_food_screen_v1/sources.json').read_text(encoding='utf-8')) if r['id']=='BV1tP4y1R7Ye');rows=[]
 for index,cat,roi,box in SPECS:
  f=s['frames'][index];assert sha(f['image'])==f['sha256'];im=Image.open(f['image']).crop(roi);p=E/f'{index:02d}.jpg';im.save(p,quality=95);a,b,c,d=box;local=[a-roi[0],b-roi[1],c-roi[0],d-roi[1]]
  rows.append(dict(id=f'{s["id"]}_{index:02d}',image=str(p),sha256=sha(p),source_group=s['id'],source_video_sha256=s['sha256'],source_frame=f['image'],source_frame_sha256=f['sha256'],seconds=f['seconds'],roi=roi,boxes_xyxy=[local],category=cat,training_eligible=False,review_status='pending_overlay'))
  ImageDraw.Draw(im).rectangle(local,outline='lime',width=3);im.save(E/f'{index:02d}_overlay.jpg')
 (E/'manifest.json').write_text(json.dumps(dict(records=rows,excluded=[dict(frame=i,reason='Visible ginger/animal mixture; category reference not sufficiently clear for this coarse-group protocol') for i in [6,7]],scope='4 manually bounded groups from1 previously viewed video, nonexpert. Locked before this comparison; manual ROI bypasses full-frame localization. Source not in recent detector incremental training; historical earlier exposure/pretraining not guaranteed absent. No independent blind accuracy.'),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
