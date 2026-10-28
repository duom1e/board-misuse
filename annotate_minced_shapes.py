import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;O=B/'minced_shape_annotation_v1'
# Coordinates in viewed 400x325 Wang crops, or 400x225 pepper frames.
WANG={10:('vegetable',[[55,105,280,250]]),12:('vegetable',[[95,145,215,255]]),16:('meat',[[160,77,259,190]]),18:('meat',[[90,45,240,180]]),20:('meat',[[100,50,259,155]]),22:('meat',[[100,50,259,155]]),24:('meat',[[90,50,255,160]]),28:('vegetable',[[105,75,265,155]]),30:('vegetable',[[125,90,205,155]]),34:('vegetable',[[85,85,172,192]]),36:('vegetable',[[80,85,185,175]])}
PEPPER={32:[[140,40,250,145],[270,5,400,80]],34:[[105,0,382,220]],36:[[105,10,350,190]],38:[[110,15,350,192]],42:[[130,40,330,190]],44:[[80,0,350,155]]}
if __name__=='__main__':
 rows=json.loads((O/'samples.json').read_text());accepted=[];excluded=[]
 for r in rows:
  sec=r['seconds'];wang=r['source_group']=='BV1Ys411L7r6';table=WANG if wang else PEPPER
  if sec not in table:excluded.append({'source':r['source_group'],'seconds':sec,'reason':'Scoop/hand-heavy removal or bowl transition/too-wide multi-food scene, not empty negative.'});continue
  cat,boxes=WANG[sec] if wang else ('vegetable',PEPPER[sec]);ref=(400,325) if wang else (400,225);im=Image.open(r['image']);w,h=im.size
  actual=[[round(a*w/ref[0]),round(b*h/ref[1]),min(w,round(c*w/ref[0])),min(h,round(d*h/ref[1]))] for a,b,c,d in boxes]
  for a,b,c,d in actual:assert 0<=a<c<=w and 0<=b<d<=h
  lp=O/'labels'/(Path(r['image']).stem+'.txt');lp.parent.mkdir(exist_ok=True);cls=int(cat=='vegetable');lp.write_text(''.join(f'{cls} {(a+c)/(2*w):.8f} {(b+d)/(2*h):.8f} {(c-a)/w:.8f} {(d-b)/h:.8f}\n' for a,b,c,d in actual),encoding='utf-8')
  accepted.append({**r,'class':cat,'boxes_xyxy':actual,'label':str(lp),'sha256':hashlib.sha256(Path(r['image']).read_bytes()).hexdigest(),'label_sha256':hashlib.sha256(lp.read_bytes()).hexdigest(),'intended_role':'training_candidate' if wang else 'reserved_new_source_validation_candidate_not_predicted','review_status':'pending_overlay'})
 for source in ['BV1Ys411L7r6','BV14K4y1x7LF']:
  rs=[r for r in accepted if r['source_group']==source];sheet=Image.new('RGB',(1600,((len(rs)+3)//4)*355),'white')
  for i,r in enumerate(rs):
   im=Image.open(r['image']).convert('RGB');dr=ImageDraw.Draw(im)
   for box in r['boxes_xyxy']:dr.rectangle(box,outline='lime',width=5)
   im.thumbnail((400,325));x=i%4*400;y=i//4*355;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+330),f'{r["seconds"]}s {r["class"]}',fill='black')
  sheet.save(O/(source+'_labels_review.jpg'))
 (O/'manifest.json').write_text(json.dumps({'records':accepted,'excluded':excluded,'review_status':'pending_overlay','scope':'Same-class visible food-group coarse boxes, no board-contact proof. Wang foreground board ROI excludes counter bowls; not exhaustive full-frame food GT. Source validation reserved separately; no random adjacent-frame split.','expert_review':False,'training_started':False,'predictions_run_on_reserved_source':False,'licensing':'User local processing authorized, creator reuse license not established. No media/weight publishing.'},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'images':len(accepted),'boxes':sum(len(r['boxes_xyxy']) for r in accepted),'excluded':len(excluded)}))
