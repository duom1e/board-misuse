import json,hashlib,shutil
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;O=B/'canteen_food_multiview_v1'
BOXES={2:[(350,200,565,430)],10:[(230,170,565,430)],15:[(225,140,565,430)],23:[(185,110,430,290)],33:[(155,142,510,280)],45:[(230,140,410,260)],55:[(260,165,420,290)],70:[(260,150,475,265)],80:[(155,100,430,270)],95:[(240,105,470,300)],105:[(165,145,430,270)],135:[],155:[(100,55,495,280)],175:[(95,50,495,260)],200:[(130,50,430,255)]}
if __name__=='__main__':
 O.mkdir(exist_ok=False);old=json.loads((B/'canteen_food_boxes_v2/manifest.json').read_text(encoding='utf-8'));rows=[]
 def add(r,name,camera):
  r=dict(r);im=Image.open(r['image']);w,h=im.size;ip=O/'images'/r['split']/(name+'.png');lp=O/'labels'/r['split']/(name+'.txt');ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r['image'],ip)
  cls=0 if r['class']=='meat' else 1
  lp.write_text(''.join(f'{cls} {(a+c)/(2*w):.8f} {(b+d)/(2*h):.8f} {(c-a)/w:.8f} {(d-b)/h:.8f}\n' for a,b,c,d in r['boxes_xyxy']),encoding='utf-8')
  r.update(image=str(ip),label=str(lp),camera=camera,sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(lp.read_bytes()).hexdigest());rows.append(r)
 for r in old['records']:add(r,'overhead_'+Path(r['image']).stem,'overhead')
 samples=json.loads((B/'canteen_side_training_samples/samples.json').read_text())
 for r in samples:
  sec=r['seconds']
  if sec==120:continue # Meat held at board edge: contact/scope ambiguous, not a negative.
  add({**r,'split':'train','class':'meat' if sec<30 else 'empty' if sec==135 else 'vegetable','boxes_xyxy':BOXES[sec]},f'side_train_{sec:03}','side')
 side=json.loads((B/'canteen_side_diagnostic/manifest.json').read_text())
 for r in side['records']:add({**r,'split':'val'},f'side_val_{r["seconds"]:03}','side')
 new=[r for r in rows if r['camera']=='side' and r['split']=='train'];sheet=Image.new('RGB',(1860,1380),'white')
 for i,r in enumerate(new):
  im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im)
  for box in r['boxes_xyxy']:d.rectangle(box,outline='lime',width=3)
  im.thumbnail((465,323));x=i%4*465;y=i//4*345;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+326),str(r['seconds'])+'s',fill='black')
 sheet.save(O/'side_train_review.jpg')
 (O/'manifest.json').write_text(json.dumps({'records':rows,'review_status':'pending_new_overlay_review','scope':'Same operation multi-view development. Side8 now used for selection, no longer unselected diagnostic. Adjacent frames and same actions strongly correlated. No independent test.','excluded_proposals':[{'seconds':120,'camera':'side','reason':'Meat held at edge; ambiguous board association; not used as empty negative.'}],'source_license':'CC BY-NC4 local research only'},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'total':len(rows),'train':sum(r['split']=='train' for r in rows),'val':sum(r['split']=='val' for r in rows)}))
