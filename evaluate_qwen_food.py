"""Fixed-prompt development diagnostic. Image categories, not detection accuracy."""
import json, hashlib, time
from pathlib import Path
B=Path(__file__).resolve().parent
E=B/'experiments/qwen3vl_frozen_baseline'
PROMPT='Classify visible food in the foreground cutting-board work area. Ignore subtitles. Do not infer raw/cooked state or hygiene compliance. Vegetable includes peppers and garlic; meat means animal flesh. Return only one word: meat, vegetable, mixed, empty, or unknown. Use unknown if visual evidence is insufficient.'
def save(name,value):
    (E/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    rows=[]
    for folder in ['minced_shape_annotation_v1','bilibili_prep_manual_v1']:
        manifest=json.loads((B/folder/'manifest.json').read_text(encoding='utf-8'))
        for row in manifest['records']:
            if row.get('localization_uncertainty')!='normal': continue
            rows.append({'image':row['image'],'expected':row['class'] if row['boxes_xyxy'] else 'empty','source':folder,'sha256':hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()})
    downloaded=json.loads((E/'download_status.json').read_text(encoding='utf-8'))
    assert downloaded['stage']=='downloaded'
    plan={'prompt':PROMPT,'rows':rows,'revision':downloaded['revision'],'sampling':False,'max_new_tokens':12,'thumbnail':512,'scope':'Consumed development images; image-level food classification only. No training, prompt search, independent test or adoption.'}
    if (E/'locked_plan.json').exists(): assert json.loads((E/'locked_plan.json').read_text(encoding='utf-8'))==plan
    else: save('locked_plan.json',plan)
    assert not (E/'results.json').exists()
    import torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from PIL import Image
    model=Qwen3VLForConditionalGeneration.from_pretrained(downloaded['folder'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(downloaded['folder'],local_files_only=True)
    results=[]
    for row in rows:
        image=Image.open(row['image']).convert('RGB'); image.thumbnail((512,512))
        messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':PROMPT}]}]
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
        start=time.time()
        with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=12,do_sample=False)
        answer=processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        predicted=answer.strip().lower().rstrip('.')
        if predicted not in ['meat','vegetable','mixed','empty','unknown']: predicted='invalid'
        results.append({**row,'response':answer,'predicted':predicted,'seconds':time.time()-start})
        save('partial_results.json',results)
        print(len(results),len(rows),row['expected'],predicted,flush=True)
    save('results.json',{'rows':results,'correct':sum(r['expected']==r['predicted'] for r in results),'total':len(results),'adopted':False,'training_performed':False})
if __name__=='__main__': main()
