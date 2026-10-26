import json,hashlib,shutil
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;A=B/'redveg_meat_source_audit';O=B/'redveg_meat_contrast_v1'
# Manual boxes in the viewed thumbnail coordinate systems, scaled to original size.
RED={0:[[22,220,151,265]],2:[[18,49,186,157]],3:[[0,65,186,160]],4:[[0,125,181,255]],5:[[28,120,186,275]],6:[[65,120,186,225]],7:[[38,142,186,255]],9:[[0,75,186,330]],10:[[60,150,186,330],[12,58,105,95]]}
MEAT={0:[[65,42,305,153]],1:[[90,94,193,158],[200,105,270,155],[260,176,398,225]],2:[[67,115,158,215],[150,145,345,225],[135,25,211,105]],3:[[120,35,320,130],[0,136,125,225],[125,174,336,225]],4:[[28,92,100,155],[89,120,164,189],[155,134,306,206]],6:[[122,58,201,112]],7:[[260,55,400,164],[105,90,177,124]]}
if __name__=='__main__':
 O.mkdir(exist_ok=True);records=[]
 for cat,file,boxes,ref in [('vegetable','redpepper_samples.json',RED,(186,330)),('meat','meat_samples.json',MEAT,(400,225))]:
  samples=json.loads((A/file).read_text(encoding='utf-8'))
  for idx,bs in boxes.items():
   row=samples[idx];im=Image.open(row['image']);w,h=im.size
   actual=[[round(a*w/ref[0]),round(b*h/ref[1]),min(w,round(c*w/ref[0])),min(h,round(d*h/ref[1]))] for a,b,c,d in bs]
   for a,b,c,d in actual:assert 0<=a<c<=w and 0<=b<d<=h
   name=f'{cat}_{row["frame"]:06}';ip=O/'images'/(name+'.jpg');lp=O/'labels'/(name+'.txt');ip.parent.mkdir(exist_ok=True);lp.parent.mkdir(exist_ok=True);shutil.copyfile(row['image'],ip)
   cls=int(cat=='vegetable');lp.write_text(''.join(f'{cls} {(a+c)/(2*w):.8f} {(b+d)/(2*h):.8f} {(c-a)/w:.8f} {(d-b)/h:.8f}\n' for a,b,c,d in actual),encoding='utf-8')
   records.append({**row,'image':str(ip),'class':cat,'boxes_xyxy':actual,'label':str(lp),'sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),'label_sha256':hashlib.sha256(lp.read_bytes()).hexdigest(),'source_group':Path(row['source_video']).stem,'sample_index':idx,'review_status':'pending_overlay','role':'development_training_candidate_not_trained'})
  rs=[r for r in records if r['class']==cat];sheet=Image.new('RGB',(1600,((len(rs)+3)//4)*360),'white')
  for j,r in enumerate(rs):
   im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im)
   for box in r['boxes_xyxy']:d.rectangle(box,outline='lime',width=5)
   im.thumbnail((400,330));x=j%4*400;y=j//4*360;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+335),f'idx{r["sample_index"]} {r["seconds"]:.1f}s',fill='black')
  sheet.save(O/(cat+'_annotation_review.jpg'))
 (O/'manifest.json').write_text(json.dumps({'records':records,'review_status':'pending_overlay','annotation':'Assistant manual visible same-class food-group boxes, not expert or instance segmentation. No auto labels from title/predictions.','scope':'Two video sources, one source per class: class and scene are confounded. Not a balanced independent benchmark. Do not random-frame split as independent train/test.','source_rights':'User local processing authorized. Creator reuse/training license not established; no publishing imagery or weights.','excluded':'Pepper knife display/presentation, heavy blade cover; meat trimming/hand-obscured, bowl mixing and recipe end card.'},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'images':len(records),'boxes':sum(len(r['boxes_xyxy']) for r in records)}))
