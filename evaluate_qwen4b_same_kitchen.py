"""Automatic coarse localization + conservative gate on four consumed development images."""
import json,hashlib
from pathlib import Path
from food_region_gate import locate
B=Path(__file__).resolve().parent;E=B/'experiments/qwen4b_same_kitchen_v1'
PROMPT='Ignore subtitles. Locate visible cutting boards, food containers (bowls, plates, jars, processor bowls), and visible food groups. Return JSON only with keys boards, containers, food; each value is a list of bounding boxes [x1,y1,x2,y2]. Coordinates must be normalized integers 0 to 1000 relative to the whole image. Do not include hands or knives as food. Empty lists if absent. Use separate boxes for separate groups; food in containers still needs a food box. Maximum 6 boxes per category.'
def save(n,d): (E/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    E.mkdir(exist_ok=True)
    manifest=B/'same_kitchen_wood_v1/manifest.json';rows=json.loads(manifest.read_text(encoding='utf-8'))['records']
    download=json.loads((B/'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    assert not (E/'locked_plan.json').exists()
    save('locked_plan.json',{'prompt':PROMPT,'revision':download['revision'],'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'thumbnail':512,'max_new_tokens':300,'do_sample':False,'scope':'Two new kitchens,2 selected neighboring positive-board frames +8 negative-counter frames, locked before inference. Fixed old prompt/settings. No independent event accuracy or training/adoption.'})
    import torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from PIL import Image,ImageDraw
    model=Qwen3VLForConditionalGeneration.from_pretrained(download['folder'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(download['folder'],local_files_only=True)
    results=[]
    for r in rows:
        original=Image.open(r['image']).convert('RGB');im=original.copy();im.thumbnail((512,512))
        msg=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':PROMPT}]}]
        inputs=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=300,do_sample=False)
        answer=processor.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        try:
            p=json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip());assert set(p)=={'boards','containers','food'}
            for boxes in p.values():
                assert isinstance(boxes,list) and len(boxes)<=6
                for box in boxes:assert len(box)==4 and all(type(v) is int and 0<=v<=1000 for v in box) and box[2]>box[0] and box[3]>box[1]
            w,h=original.size;pixel={k:[[x1*w/1000,y1*h/1000,x2*w/1000,y2*h/1000] for x1,y1,x2,y2 in v] for k,v in p.items()}
            gated=[{'box':box,**locate(box,pixel['boards'],pixel['containers'])} for box in pixel['food']]
            dr=ImageDraw.Draw(original)
            for k,color in [('boards','cyan'),('containers','yellow'),('food','red')]:
                for box in pixel[k]:dr.rectangle(box,outline=color,width=4);dr.text(box[:2],k,fill=color)
            original.save(E/(r['case']+'_prediction.jpg'))
            result={'case':r['case'],'response':answer,'pixel_boxes':pixel,'gated':gated,'status':'needs_review'}
        except Exception as exc:result={'case':r['case'],'response':answer,'parse_error':repr(exc),'status':'needs_review'}
        results.append(result);save('partial_results.json',results);print(r['case'],result.get('gated',result.get('parse_error')),flush=True)
    save('results.json',{'rows':results,'adopted':False,'training_performed':False})
if __name__=='__main__':main()
