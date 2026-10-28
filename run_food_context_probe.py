"""Targeted development check with stored local opinions and new context opinions."""
import json,hashlib
from pathlib import Path
from PIL import Image
from food_context_review import CONTEXT_PROMPT,review_food_context
from region_vlm_runtime import LocalRegionModel
B=Path(__file__).resolve().parent;E=B/'experiments/food_context_probe_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):(E/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    E.mkdir(exist_ok=True);assert not (E/'locked_plan.json').exists()
    local_file=B/'food_multiclass_probe_v1/results.json';manifest=B/'food_multiclass_probe_v1/manifest.json';carrot_file=B/'experiments/carrot_video_v1/live_run/results.json'
    labels={r['case']:r for r in json.loads(manifest.read_text(encoding='utf-8'))['records']};groups={}
    for r in json.loads(local_file.read_text(encoding='utf-8'))['rows']:
        label=labels[r['case']];groups.setdefault(label['source_image'],[]).append({'case':r['case'],'food_category':r['predicted'],'category_inference_performed':r['inference_performed'],'local_raw_response':r['raw_response'],'expected_local':label['expected'],'local_scored':r['scored']})
    for r in json.loads(carrot_file.read_text(encoding='utf-8'))['rows']:
        if r['frame_seconds']!=50:continue
        groups[r['image']]=[dict(f,case=f'carrot50_{i}',expected_local='vegetable',local_scored=False) for i,f in enumerate(r['food_regions']) if f.get('category_inference_performed')]
    meta=json.loads((B/'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    save('locked_plan.json',{'context_prompt':CONTEXT_PROMPT,'model_revision':meta['revision'],'thumbnail':512,'max_new_tokens':12,'do_sample':False,'inputs':[{'image':image,'sha256':sha(Path(image)),'local_opinions':rows} for image,rows in groups.items()],'historical_files':{str(p):sha(p) for p in [local_file,manifest,carrot_file]},'scope':'Targeted posthoc development diagnosis,21 previously consumed local opinions in9 source frames. SAME model, separate full-image question, not independent evidence, automatic correction or new blind accuracy. No training/adoption.'})
    model=LocalRegionModel();results=[]
    for image,rows in groups.items():
        enriched,context=review_food_context(Image.open(image).convert('RGB'),rows,lambda im:model.predict(im,prompt=CONTEXT_PROMPT,max_new_tokens=12))
        results.append({'image':image,'context_review':context,'regions':enriched});save('partial_results.json',{'rows':results});print(Path(image).name,context.get('context_category'),context.get('contradiction_count'),flush=True)
    save('results.json',{'rows':results,'adopted':False,'training_performed':False,'local_categories_replaced':False})
    plan=json.loads((E/'locked_plan.json').read_text(encoding='utf-8'))
    for p,digest in plan['historical_files'].items():assert sha(Path(p))==digest
if __name__=='__main__':main()
