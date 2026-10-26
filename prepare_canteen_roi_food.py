import cv2,json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;OUT=B/'canteen_roi_food_v1';VIDEO=B.parent/'work/datasets/cn_commercial_kitchen/overhead_cutting.mp4'
# Conservative visible-food phase proposals, to be visually reviewed before training.
PLAN={'train':{'meat':[10,15,20,25,30],'vegetable':[93,98,103,108,113,138,143,148,153,239,244,249,254],'empty':[190,194,198]},
      'val':{'meat':[45,50,55],'vegetable':[168,173,178],'empty':[208,212]},
      'test':{'meat':[65,70,74],'vegetable':[275,280,285],'empty':[222,226]}}
ROI=[.389,.354,.715,.825]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT.mkdir(exist_ok=False)
(OUT/'sampling_plan.json').write_text(json.dumps({'video':str(VIDEO),'video_sha256':digest(VIDEO),'roi':ROI,'times_seconds':PLAN,'scope':'Same-video fixed-ROI positive control only, not independent or complete misuse evaluation.','empty_definition':'No obvious bulk food in board ROI; not cleanliness, germs or small-residue certification.','label_provenance':'Time proposals followed by assistant visual review, not expert food labels.'},ensure_ascii=False,indent=2),encoding='utf-8')
cap=cv2.VideoCapture(str(VIDEO));fps=cap.get(cv2.CAP_PROP_FPS);records=[]
for split,classes in PLAN.items():
 for category,times in classes.items():
  for sec in times:
   frame=round(sec*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,array=cap.read()
   if not ok:raise RuntimeError('Could not decode selected frame')
   height,width=array.shape[:2];box=tuple(round(v*(width if i%2==0 else height)) for i,v in enumerate(ROI))
   im=Image.fromarray(cv2.cvtColor(array,cv2.COLOR_BGR2RGB)).crop(box)
   path=OUT/split/category/f'{category}_{sec:03d}.png';path.parent.mkdir(parents=True,exist_ok=True);im.save(path)
   records.append({'split':split,'class':category,'seconds':sec,'frame':frame,'image':str(path),'sha256':digest(path),'source_video':str(VIDEO),'crop_xyxy':box})
cap.release()
for category in ['meat','vegetable','empty']:
 rows=[r for r in records if r['class']==category];sheet=Image.new('RGB',(1280,((len(rows)+3)//4)*245),'white')
 for i,r in enumerate(rows):
  im=Image.open(r['image']).convert('RGB');im.thumbnail((315,215));x=i%4*320;y=i//4*245;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x,y+220),f'{r["split"]} {category} {r["seconds"]}s',fill='black')
 sheet.save(OUT/(category+'_review.jpg'))
(OUT/'manifest.json').write_text(json.dumps({'records':records,'review_status':'pending_visual_review','same_source_group':True,'not_fresh_independent_test':True},ensure_ascii=False,indent=2),encoding='utf-8')
print(len(records))
