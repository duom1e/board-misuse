"""New-source fixed-prompt categories and predeclared subtitle-mask control."""
import json,hashlib,time
from pathlib import Path
from evaluate_qwen_food import PROMPT
B=Path(__file__).resolve().parent
E=B/'experiments/qwen_new_sources_v1'
def save(name,value):
    (E/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    rows=[]
    specs=[('chicken_BV1264y1F7a6',range(10),'meat'),('boardcare_BV1Bt411Y7VF',[2,7,9],'empty'),('boardcare_BV1Bt411Y7VF',[3,4],'unknown')]
    for source,indices,label in specs:
        for i in indices:
            image=E/f'{source}_{i:02}.jpg'
            rows.append({'image':str(image),'source':source,'index':i,'expected':label,'sha256':hashlib.sha256(image.read_bytes()).hexdigest()})
    downloaded=json.loads((B/'experiments/qwen3vl_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    plan={'prompt':PROMPT,'revision':downloaded['revision'],'rows':rows,'variants':['original','mask_bottom_15_percent'],'max_new_tokens':12,'do_sample':False,'thumbnail':512,'labels':'assistant visual review before predictions, not expert','exclusions':'Chicken10 motion/occlusion and11 serving bowl; boardcare presenter/washing/graphic/vertical-board views excluded. Basin3/4 predeclared unknown: no cutting board visible.','scope':'Two new project video sources, assistant-selected correlated frames. Public-pretraining overlap unknown. No vegetable source this round. Image categories, NOT boxes or misuse accuracy. No training/adoption.','source_urls':['https://www.bilibili.com/video/BV1264y1F7a6/','https://www.bilibili.com/video/BV1Bt411Y7VF/'],'reuse':'Local research only; creator training/republication permission not established.'}
    if (E/'locked_plan.json').exists():assert json.loads((E/'locked_plan.json').read_text(encoding='utf-8'))==plan
    else:save('locked_plan.json',plan)
    assert not (E/'results.json').exists()
    import torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from PIL import Image,ImageDraw
    model=Qwen3VLForConditionalGeneration.from_pretrained(downloaded['folder'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(downloaded['folder'],local_files_only=True)
    results=[]
    for variant in plan['variants']:
        for row in rows:
            image=Image.open(row['image']).convert('RGB')
            if variant!='original':ImageDraw.Draw(image).rectangle((0,int(image.height*.85),image.width,image.height),fill=(127,127,127))
            image.thumbnail((512,512))
            messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':PROMPT}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
            start=time.time()
            with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=12,do_sample=False)
            answer=processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            prediction=answer.strip().lower().rstrip('.')
            if prediction not in ['meat','vegetable','mixed','empty','unknown']:prediction='invalid'
            results.append({**row,'variant':variant,'predicted':prediction,'response':answer,'seconds':time.time()-start})
            save('partial_results.json',results);print(len(results),30,row['expected'],prediction,flush=True)
    save('results.json',{'rows':results,'training_performed':False,'adopted':False})
if __name__=='__main__':main()
