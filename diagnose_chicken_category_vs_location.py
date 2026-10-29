"""Separate proposal geometry from label errors; VLM opinions never overwrite."""
import json,hashlib,os
from pathlib import Path
from PIL import Image,ImageDraw
from evaluate_food_group_boxes import iou
B=Path(__file__).resolve().parent;E=B/'chicken_error_decomposition_v1'
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 E.mkdir(exist_ok=False);data=json.loads((B/'food_group_boxes_v1/manifest.json').read_text(encoding='utf-8'))['records'];gt={r['id']:r for r in data};rows=json.loads((B/'food_yellow_chicken_adaptation_v1/group_results.json').read_text(encoding='utf-8'));held=[r for r in rows if r['split']=='source_held_development'];matches=[];cases=[]
 for r in held:
  used=set()
  for j,box in enumerate(gt[r['id']]['boxes_xyxy']):
   scores=[(iou(box,q['box']),i) for i,q in enumerate(r['predictions']) if i not in used and iou(box,q['box'])>=.5]
   if scores:
    score,k=max(scores);used.add(k);matches.append(dict(id=r['id'],gt_index=j,pred_index=k,iou=score,pred_category=r['predictions'][k]['category'],reference='meat'))
  for k,q in enumerate(r['predictions']):cases.append(dict(id=r['id'],pred_index=k,image=gt[r['id']]['image'],source_sha256=gt[r['id']]['sha256'],proposal=q))
 save('locked_plan.json',dict(cases=cases,matches=matches,scope='Two consumed held-source ROIs, six predictions selected after known failures; diagnostic only, no blind test or automatic relabeling. Category-agnostic matching ignores label but remains one-to-one IoU>=.5.'))
 from region_vlm_runtime import LocalRegionModel
 from food_category_review import FOOD_CATEGORY_PROMPT,review_board_foods
 model=LocalRegionModel();out=[]
 for c in cases:
  assert hashlib.sha256(Path(c['image']).read_bytes()).hexdigest()==c['source_sha256']
  im=Image.open(c['image']).convert('RGB');q=review_board_foods(im,[dict(box=c['proposal']['box'],location='board_candidate')],lambda crop:model.predict(crop,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12))[0]
  q.pop('location');c['review']=q;c['board_relation_evaluated']=False;out.append(c);save('partial_results.json',out)
  crop=im.crop(q['category_crop_xyxy']);crop.save(E/f'{c["id"]}_{c["pred_index"]}.jpg');print(c['id'],c['pred_index'],c['proposal']['category'],q['food_category'],flush=True)
 save('results.json',dict(rows=out,matches=matches,training_performed=False,adopted=False))
 sheet=Image.new('RGB',(1200,850),'white');d=ImageDraw.Draw(sheet)
 for i,c in enumerate(out):
  im=Image.open(E/f'{c["id"]}_{c["pred_index"]}.jpg');im.thumbnail((390,340));x=i%3*400;y=i//3*425;sheet.paste(im,(x,y));d.text((x,y+345),str(c['pred_index'])+' '+c['proposal']['category']+' -> '+c['review']['food_category'],fill='black')
 sheet.save(E/'review.jpg')
if __name__=='__main__':main()
