"""Development-only board presence and visible food-support relation diagnostic."""
import json,hashlib,time
from pathlib import Path
B=Path(__file__).resolve().parent
E=B/'experiments/qwen_board_relation_v1'
PROMPT='Inspect the image, ignoring subtitles. Is a cutting board visibly present? Do not treat a tabletop, bowl or wash basin as a cutting board. Then inspect food visibly supported by the board surface. Food inside a bowl does not count as food on the board. Do not infer hygiene compliance or raw/cooked status. Return JSON only with exactly these keys: "board": "present|absent|uncertain", "food_on_board": "meat|vegetable|mixed|none|not_applicable|uncertain". If board is absent, food_on_board must be not_applicable. If evidence is occluded or unclear use uncertain.'
def save(name,data):
    (E/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    E.mkdir(parents=True,exist_ok=True)
    previous=json.loads((B/'experiments/qwen_new_sources_v1/locked_plan.json').read_text(encoding='utf-8'))
    rows=[]
    for r in previous['rows']:
        rows.append({**r,'expected_board':'absent' if r['expected']=='unknown' else 'present','expected_food_on_board':{'meat':'meat','empty':'none','unknown':'not_applicable'}[r['expected']]})
    downloaded=json.loads((B/'experiments/qwen3vl_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    plan={'prompt':PROMPT,'revision':downloaded['revision'],'rows':rows,'max_new_tokens':80,'do_sample':False,'thumbnail':512,'scope':'Same consumed 15 development frames, new task/prompt. Assistant labels before predictions. Not new-source independent accuracy. No boxes or contact/event proof; no training/adoption.'}
    if (E/'locked_plan.json').exists():assert json.loads((E/'locked_plan.json').read_text(encoding='utf-8'))==plan
    else:save('locked_plan.json',plan)
    assert not (E/'results.json').exists()
    import torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from PIL import Image
    model=Qwen3VLForConditionalGeneration.from_pretrained(downloaded['folder'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(downloaded['folder'],local_files_only=True)
    results=[]
    for r in rows:
        assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==r['sha256']
        im=Image.open(r['image']).convert('RGB');im.thumbnail((512,512))
        msg=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':PROMPT}]}]
        inputs=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
        start=time.time()
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=80,do_sample=False)
        answer=processor.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        try:
            parsed=json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip())
            assert set(parsed)=={'board','food_on_board'}
            assert parsed['board'] in ['present','absent','uncertain']
            assert parsed['food_on_board'] in ['meat','vegetable','mixed','none','not_applicable','uncertain']
            assert parsed['board']!='absent' or parsed['food_on_board']=='not_applicable'
        except Exception:parsed={'board':'invalid','food_on_board':'invalid'}
        results.append({**r,'response':answer,'prediction':parsed,'seconds':time.time()-start,'review_status':'needs_review'})
        save('partial_results.json',results);print(len(results),len(rows),parsed,flush=True)
    save('results.json',{'rows':results,'adopted':False,'training_performed':False})
if __name__=='__main__':main()
