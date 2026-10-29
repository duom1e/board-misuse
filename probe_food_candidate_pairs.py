"""Real tight/10%-expanded comparison on four consumed detector proposals."""
import json
from pathlib import Path
from PIL import Image
from region_vlm_runtime import LocalRegionModel
from food_category_review import FOOD_CATEGORY_PROMPT,review_board_foods
B=Path(__file__).resolve().parent;E=B/'food_candidate_pairs_v1'
def main():
 E.mkdir(exist_ok=False)
 source=B/'food_batch36_v2';suspects=json.loads((source/'yolo_suspects.json').read_text(encoding='utf-8'));cases=json.loads((source/'locked_plan.json').read_text(encoding='utf-8'))['cases']
 (E/'plan.json').write_text(json.dumps(dict(suspects=suspects,scope='Four already consumed error-selected YOLO proposals. Internal board_candidate only enables crop helper; no spatial or board-contact ground truth.',prompt=FOOD_CATEGORY_PROMPT),ensure_ascii=False,indent=2),encoding='utf-8')
 model=LocalRegionModel();rows=[]
 for s in suspects:
  c=cases[s['index']];image=Image.open(c['image']).convert('RGB')
  r=review_board_foods(image,[dict(box=s['selected']['box'],location='board_candidate')],lambda crop:model.predict(crop,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12),compare_tight_crop=True)[0]
  r.pop('location');r.update(index=s['index'],source=c['source'],original_yolo=s['selected'],spatial_relation_not_evaluated=True,compliance_determined=False)
  rows.append(r);(E/'partial_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');print(s['index'],r['food_category'],r['tight_crop_comparison']['tight_opinion'],flush=True)
 (E/'results.json').write_text(json.dumps(dict(rows=rows,training_performed=False,adopted=False),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
