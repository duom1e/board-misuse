"""Manual foreground-board region labels; no prediction-generated truth."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;O=B/'bilibili_prep_manual_v1'
# Coordinates in each 580x285 crop, assigned after viewing actual crop sheets.
FOOD={1195:[100,70,455,235],1225:[265,115,407,215],1255:[265,80,515,225],1285:[295,135,540,225],1315:[295,70,480,210],1345:[285,40,455,160],1375:[260,95,465,230],1405:[255,100,375,175],1435:[225,80,385,150],1465:[225,55,400,175],1495:[225,55,400,175],1855:[145,35,380,175],1885:[145,35,365,175],1915:[145,35,400,175],1945:[140,35,380,170],1975:[140,35,405,170],2005:[140,35,270,170],2035:[140,35,355,175],2065:[140,35,370,175],2095:[140,35,455,205],2125:[140,35,455,205],2155:[140,35,470,200],2185:[140,35,465,195]}
if __name__=='__main__':
 rows=json.loads((O/'crop_candidates.json').read_text(encoding='utf-8'))
 for r in rows:
  f=r['frame'];r['boxes_xyxy']=[FOOD[f]] if f in FOOD else [];r['class']='vegetable' if f in FOOD else 'empty'
  r['board_box_xyxy']=[20 if f==1195 else 25,20,525,275]
  r['box_annotation_status']='pending_overlay_review';r['localization_uncertainty']='high' if f in [1225,1285,1315,1345,1375] else 'normal'
  r['image']=r['crop_image'];r['sha256']=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()
  for folder,boxes,cls in [('food_labels',r['boxes_xyxy'],1),('board_labels',[r['board_box_xyxy']],0)]:
   p=O/folder/f'crop_{f:06}.txt';p.parent.mkdir(exist_ok=True)
   for a,b,c,d in boxes:assert 0<=a<c<=580 and 0<=b<d<=285
   p.write_text(''.join(f'{cls} {(a+c)/1160:.8f} {(b+d)/570:.8f} {(c-a)/580:.8f} {(d-b)/285:.8f}\n' for a,b,c,d in boxes),encoding='utf-8')
  r['annotation']='assistant_manual_food_group_with_provisional_board_crop_envelope_not_expert'
  r['board_box_status']='provisional_not_ground_truth_do_not_train'
 for start in [0,12,24]:
  rs=rows[start:start+12];sheet=Image.new('RGB',(1740,((len(rs)+2)//3)*315),'white')
  for i,r in enumerate(rs):
   im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im);d.rectangle(r['board_box_xyxy'],outline='cyan',width=2)
   for box in r['boxes_xyxy']:d.rectangle(box,outline='lime',width=3)
   x=i%3*580;y=i//3*315;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+290),f'{r["frame"]} {r["localization_uncertainty"]}',fill='black')
  sheet.save(O/f'labels_review_{start//12+1}.jpg')
 (O/'manifest.json').write_text(json.dumps({'records':rows,'review_status':'pending_overlay_review','board_annotation_status':'provisional_common_crop_envelope_not_ground_truth_do_not_train','scope':'Foreground board crop only. Other full-frame boards/food not exhaustive. Board boxes are common crop proposals, NOT verified board ground truth. Same-source exposed diagnostic; not independent final benchmark.','expert_review':False,'rights':'User local processing authorized; creator training license not established.','high_occlusion_policy':'High-uncertainty group boxes retained for review but exclude from strict localization probe.','empty_definition':'No obvious bulk food, seeds/residue may remain; not clean or compliant.'},ensure_ascii=False,indent=2),encoding='utf-8')
 print(len(rows))
