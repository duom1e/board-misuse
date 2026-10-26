import json,shutil,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;O=B/'canteen_food_boxes_v2'
# Pixel boxes manually assigned after viewing the 417x339 source crops.
BOXES={35:[(42,26,201,142)],40:[(35,26,201,143)],58:[(35,26,201,143)],60:[(76,20,153,85)],62:[(144,120,177,140)],67:[(93,86,241,166)],72:[(78,70,243,159)],76:[(173,43,312,105)],115:[(80,109,221,185)],118:[(83,107,222,191)],155:[(92,65,218,181)],160:[(75,50,227,184)],258:[(56,52,350,186)],263:[(100,43,340,196)],268:[(99,55,352,213)],273:[(78,66,365,223)]}
if __name__=='__main__':
 O.mkdir(exist_ok=False)
 old=json.loads((B/'canteen_food_boxes_v1/manifest.json').read_text(encoding='utf-8'));rows=[]
 for r in old['records']:
  r=dict(r)
  for k,folder,ext in [('image','images','.png'),('label','labels','.txt')]:
   dst=O/folder/r['split']/(Path(r[k]).stem+ext);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r[k],dst);r[k]=str(dst)
  rows.append(r)
 new=[]
 for sec,boxes in BOXES.items():
  cls=0 if sec<80 else 1;name=f'shape_{sec:03}';src=B/'canteen_shape_samples'/f'{sec:03}.png'
  ip=O/'images/train'/(name+'.png');lp=O/'labels/train'/(name+'.txt');shutil.copyfile(src,ip)
  lp.write_text(''.join(f'{cls} {(a+c)/834:.8f} {(b+d)/678:.8f} {(c-a)/417:.8f} {(d-b)/339:.8f}\n' for a,b,c,d in boxes),encoding='utf-8')
  r={'split':'train','class':['meat','vegetable'][cls],'seconds':sec,'frame':sec*15,'image':str(ip),'label':str(lp),'boxes_xyxy':boxes,'sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),'label_sha256':hashlib.sha256(lp.read_bytes()).hexdigest(),'source_video':old['records'][0]['source_video'],'crop_xyxy':[498,255,915,594],'annotation':'assistant_manual_coarse_visible_food_group','expert_review':False};rows.append(r);new.append(r)
 sheet=Image.new('RGB',(1668,1476),'white')
 for i,r in enumerate(new):
  im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im)
  for box in r['boxes_xyxy']:d.rectangle(box,outline='lime',width=3)
  x=i%4*417;y=i//4*369;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+344),str(r['seconds'])+'s',fill='black')
 sheet.save(O/'new_annotations_review.jpg')
 (O/'manifest.json').write_text(json.dumps({**old,'records':rows,'review_status':'pending_new_overlay_review','new_training_images':16,'scope':'Same-video shape coverage development experiment only. New frames neighbor previously exposed validation/test; strong temporal leakage acknowledged, NOT independent performance.'},ensure_ascii=False,indent=2),encoding='utf-8')
 (O/'data.yaml').write_text(f'path: {O.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: meat\n  1: vegetable\n',encoding='utf-8')
 print(len(rows))
