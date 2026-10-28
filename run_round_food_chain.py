import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
from region_vlm_runtime import LocalRegionModel
from evaluate_qwen4b_regions import PROMPT
from region_response_validation import validate_region_response
from food_region_gate import locate
from food_category_review import review_board_foods,FOOD_CATEGORY_PROMPT
B=Path(__file__).resolve().parent;E=B/'round_food_chain_v1'
def save(n,o):(E/n).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 E.mkdir(exist_ok=True);assert not (E/'locked_plan.json').exists()
 cases=json.loads((B/'round_board_sources_v1/locked_boxes.json').read_text(encoding='utf-8'))['records'];save('locked_plan.json',dict(cases=cases,region_prompt=PROMPT,category_prompt=FOOD_CATEGORY_PROMPT,scope='Six consumed development images with clear chicken; actual automatic regions, not manual food boxes. No training/contact/compliance truth.'))
 model=LocalRegionModel();rows=[]
 for i,r in enumerate(cases):
  p=Path(r['image']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];im=Image.open(p).convert('RGB');raw=model.predict(im,prompt=PROMPT,max_new_tokens=300);v=validate_region_response(raw);regions=[]
  if v['valid']:
   w,h=im.size;boxes={k:[[a*w/1000,b*h/1000,c*w/1000,d*h/1000] for a,b,c,d in x] for k,x in v['normalized_boxes'].items()}
   regions=[dict(box=q,**locate(q,boxes['boards'],boxes['containers'])) for q in boxes['food']]
   regions=review_board_foods(im,regions,lambda crop:model.predict(crop,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12))
   draw=ImageDraw.Draw(im)
   for q in boxes['boards']:draw.rectangle(q,outline='cyan',width=3)
   for j,q in enumerate(regions):
    draw.rectangle(q['box'],outline='red',width=3)
    if 'category_crop_xyxy' in q:Image.open(p).crop(q['category_crop_xyxy']).save(E/f'{i:02}_{j:02}_crop.jpg')
  im.save(E/f'{i:02}_review.jpg');rows.append(dict(image=str(p),raw_response=raw,validation=v,regions=regions));save('partial_results.json',dict(rows=rows));print(i,v['valid'],[(q['location'],q['food_category']) for q in regions],flush=True)
 save('results.json',dict(rows=rows,training_performed=False,adopted=False))
if __name__=='__main__':main()
