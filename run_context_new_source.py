"""New-source manual-ROI local/context review; no detector or model selection."""
import json,hashlib
from pathlib import Path
from PIL import Image
from region_vlm_runtime import LocalRegionModel
from food_category_review import FOOD_CATEGORY_PROMPT,review_board_foods
from food_context_review import CONTEXT_PROMPT,review_food_context
B=Path(__file__).resolve().parent;E=B/'context_new_source_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):(E/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    manifest=E/'manifest.json';data=json.loads(manifest.read_text(encoding='utf-8'));meta=json.loads((B/'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    assert not (E/'locked_plan.json').exists()
    save('locked_plan.json',{'manifest_sha256':sha(manifest),'model_revision':meta['revision'],'local_prompt':FOOD_CATEGORY_PROMPT,'context_prompt':CONTEXT_PROMPT,'max_new_tokens':12,'thumbnail':512,'do_sample':False,'crop_margin':.1,'min_crop_side_pixels':20,'scope':data['scope'],'input_mode':'Human boxes invoke category stage only, not asserted board location. No training/threshold tuning/automatic correction. Meat-evidence retained when local opinion is meat or mixed; unknown is abstention, not evidence of no meat.'})
    model=LocalRegionModel();groups={}
    for r in data['records']:groups.setdefault(r['source_image'],[]).append(r)
    results=[]
    for image,records in groups.items():
        im=Image.open(image).convert('RGB');regions=[]
        for r in records:
            assert sha(Path(image))==r['source_sha256']
            region={'box':r['box_xyxy'],'location':'board_candidate','status':'needs_review','compliance_determined':False}
            local=review_board_foods(im,[region],lambda crop:model.predict(crop,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12))[0]
            assert local['category_crop_xyxy']==r['category_crop_xyxy']
            local.update(case=r['case'],expected=r['expected'],scored=r['scored'],expected_meat_evidence=r['expected_meat_evidence']);regions.append(local)
        enriched,context=review_food_context(im,regions,lambda full:model.predict(full,prompt=CONTEXT_PROMPT,max_new_tokens=12))
        results.append({'image':image,'context_review':context,'regions':enriched});save('partial_results.json',{'rows':results})
        print(Path(image).name,context['context_category'],[(r['case'],r['food_category'],r.get('category_context_comparison',{}).get('relation')) for r in enriched],flush=True)
    save('results.json',{'rows':results,'adopted':False,'training_performed':False,'local_categories_replaced':False})
if __name__=='__main__':main()
