"""Explicit dominant-food support diagnostic on consumed development frames."""
import json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
E=B/'experiments/qwen_food_support_v1'
PROMPT='Ignore subtitles. Identify what directly supports the dominant visible foreground food group. A bowl or plate remains container even if the container is on a cutting board. Do not count the board underneath a container. Select all applicable labels from board, container, hand, tabletop, uncertain. Return JSON only: {"support": ["label"]}. Do not infer food safety or whether an occluded board is empty.'
def save(n,d): (E/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    E.mkdir(exist_ok=True)
    data=B/'food_support_relation_v1/manifest.json';rows=json.loads(data.read_text(encoding='utf-8'))['records']
    download=json.loads((B/'experiments/qwen3vl_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    plan={'prompt':PROMPT,'manifest_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'revision':download['revision'],'sampling':False,'max_new_tokens':50,'thumbnail':512,'scope':'New support-label task, same consumed development frames. No prompt search, training or adoption. Not independent accuracy.'}
    assert not (E/'locked_plan.json').exists(),'Already planned; do not rerun'
    save('locked_plan.json',plan)
    import torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from PIL import Image
    model=Qwen3VLForConditionalGeneration.from_pretrained(download['folder'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(download['folder'],local_files_only=True)
    results=[]
    for r in rows:
        im=Image.open(r['image']).convert('RGB');im.thumbnail((512,512))
        messages=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':PROMPT}]}]
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=50,do_sample=False)
        answer=processor.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        try:
            p=json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip())
            assert set(p)=={'support'} and isinstance(p['support'],list) and p['support']
            assert all(v in ['board','container','hand','tabletop','uncertain'] for v in p['support'])
            predicted=sorted(set(p['support']))
        except Exception:predicted=['invalid']
        results.append({'case':r['case'],'image':r['image'],'expected':sorted(r['foreground_food_support']),'predicted':predicted,'response':answer})
        save('partial_results.json',results);print(r['case'],predicted,flush=True)
    save('results.json',{'rows':results,'strict_set_matches':sum(r['predicted']==r['expected'] for r in results),'total':len(results),'adopted':False})
if __name__=='__main__':main()
