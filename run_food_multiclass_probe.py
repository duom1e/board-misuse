"""Fixed classifier-stage diagnostic on human-provided ROIs, not end-to-end detection."""
import json,hashlib,collections
from pathlib import Path
from PIL import Image
from region_vlm_runtime import LocalRegionModel
from food_category_review import FOOD_CATEGORY_PROMPT,review_board_foods
B=Path(__file__).resolve().parent;E=B/'food_multiclass_probe_v1'
def save(name,data):(E/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    manifest=E/'manifest.json';data=json.loads(manifest.read_text(encoding='utf-8'));meta=json.loads((B/'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    assert not (E/'locked_plan.json').exists()
    save('locked_plan.json',{'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'model_revision':meta['revision'],'prompt':FOOD_CATEGORY_PROMPT,'thumbnail':512,'max_new_tokens':12,'do_sample':False,'crop_margin':.1,'min_crop_side_pixels':20,'scope':data['scope'],'input_mode':'Human ROI fed to category helper; internal board_candidate flag only enables category task, NOT a board/contact ground-truth label. No training/adoption.'})
    model=LocalRegionModel();rows=[]
    for r in data['records']:
        assert hashlib.sha256(Path(r['source_image']).read_bytes()).hexdigest()==r['source_sha256']
        region={'box':r['box_xyxy'],'location':'board_candidate','status':'needs_review','compliance_determined':False}
        out=review_board_foods(Image.open(r['source_image']).convert('RGB'),[region],lambda crop:model.predict(crop,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12))[0]
        assert out.get('category_crop_xyxy')==r['category_crop_xyxy']
        rows.append({'case':r['case'],'expected':r['expected'],'scored':r['scored'],'predicted':out['food_category'],'raw_response':out.get('category_raw_response'),'category_review_status':out['category_review_status'],'inference_performed':out['category_inference_performed'],'category_crop_xyxy':out['category_crop_xyxy']})
        save('partial_results.json',{'rows':rows});print(r['case'],r['expected'],out['food_category'],flush=True)
    scored=[r for r in rows if r['scored']];summary={c:{'matches':sum(r['predicted']==c for r in scored if r['expected']==c),'count':sum(r['expected']==c for r in scored)} for c in ['meat','vegetable','mixed','unknown']}
    save('results.json',{'rows':rows,'summary':summary,'adopted':False,'training_performed':False,'scope':'15 manually supplied, correlated scored crops; 2 predeclared unscored stress crops. Not detection or independent deployment accuracy.'});save('status.json',{'stage':'completed','adopted':False,'summary':summary})
if __name__=='__main__':main()
