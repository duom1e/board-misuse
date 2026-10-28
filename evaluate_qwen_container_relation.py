"""Development-only board presence and visible food-support relation diagnostic."""
import json,hashlib,time
from pathlib import Path
B=Path(__file__).resolve().parent
E=B/'experiments/qwen_container_relation_v1'
PROMPT='Inspect the image, ignoring subtitles. Is a cutting board visibly present? Do not treat a tabletop, bowl or wash basin as a cutting board. Then inspect food visibly supported by the board surface. Food inside a bowl does not count as food on the board. Do not infer hygiene compliance or raw/cooked status. Return JSON only with exactly these keys: "board": "present|absent|uncertain", "food_on_board": "meat|vegetable|mixed|none|not_applicable|uncertain". If board is absent, food_on_board must be not_applicable. If evidence is occluded or unclear use uncertain.'
def save(name,data):
    (E/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    E.mkdir(parents=True,exist_ok=True)
    specs=[]
    for index,frame in [(3,2461),(4,3164),(5,3868),(6,4571),(7,5274),(8,5978),(10,7384),(12,8791)]:
        specs.append((B/'minced_shape_sources_20261027'/f'pepper_{frame:06}.jpg','present' if index==4 else 'absent','none' if index==4 else 'not_applicable',f'pepper_container_{index}'))
    specs.append((B/'experiments/qwen_new_sources_v1/chicken_BV1264y1F7a6_11.jpg','present','none','chicken_serving_bowl'))
    specs.append((B/'experiments/qwen_new_sources_v1/chicken_BV1264y1F7a6_05.jpg','present','meat','consumed_positive_meat'))
    first=json.loads((B/'minced_shape_annotation_v1/manifest.json').read_text(encoding='utf-8'))['records'][0]
    specs.append((Path(first['image']),'present','vegetable','consumed_positive_vegetable'))
    rows=[{'image':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'expected_board':board,'expected_food_on_board':food,'case':case} for p,board,food,case in specs]
    downloaded=json.loads((B/'experiments/qwen3vl_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    plan={'prompt':PROMPT,'revision':downloaded['revision'],'rows':rows,'max_new_tokens':80,'do_sample':False,'thumbnail':512,'scope':'Fixed previous prompt. Eight container pepper frames, one serving bowl frame, two consumed positive controls. Existing project sources; new relation-task diagnostic, not independent final test. Assistant visual labels locked before predictions. No wooden-table negative this round. No boxes, contact/event proof, training or adoption.'}
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
