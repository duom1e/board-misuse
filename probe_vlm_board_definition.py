from pathlib import Path
import json,hashlib
import torch
from PIL import Image
import review_combined as rc
B=Path(__file__).resolve().parent

def main():
 O=B/'experiments/vlm_board_definition_probe_v1';O.mkdir(exist_ok=False)
 manifest=json.loads((B/'experiments/fusion_locked_regression_v1/manifest.json').read_text(encoding='utf-8'))
 original=rc.BOARD_PROMPT.read_text(encoding='utf-8')
 expanded=original.replace('wooden or plastic board used for cutting food', 'wooden or plastic cutting surface, including thick round chopping blocks and dark worn boards partly covered by food, hands or knives').replace('tables, counters, sinks, trays or containers', 'tables, metal counters, sinks, trays, containers, pans or woks')
 for ident,image,expected in [('wood',B/'experiments/new_source_video_v4/frames/BV1bqsfzbELh_335s.jpg',True),('wok',B/'operation_expansion_screen_v1/BV1dC7H6PEoG_03.jpg',False)]:
  manifest['cases'].append({'id':ident,'image':str(image),'expected_board_present':expected})
 (O/'plan.json').write_text(json.dumps({'cases':manifest['cases'],'prompts':{'original':original,'expanded':expanded},'limits':'previously viewed development cases; not blind; same512 input'},ensure_ascii=False,indent=2),encoding='utf-8')
 vlm=rc.load_vlm();rows=[]
 for variant,prompt in [('original',original),('expanded',expanded)]:
  limit=512
  for case in manifest['cases']:
   image=Image.open(case['image']).convert('RGB');image.thumbnail((limit,limit))
   messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
   inputs=vlm['processor'].apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(vlm['model'].device)
   with torch.inference_mode():generated=vlm['model'].generate(**inputs,max_new_tokens=200,do_sample=False)
   raw=vlm['processor'].batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
   boxes,error=rc.parse_vlm_boards(raw)
   validation=rc.validate_region_response(json.dumps({'boards':boxes,'containers':[],'food':[]})) if boxes is not None else None
   row={'variant':variant,'id':case['id'],'expected':case['expected_board_present'],'sha256':hashlib.sha256(Path(case['image']).read_bytes()).hexdigest(),'limit':limit,'actual_size':image.size,'grid':inputs['image_grid_thw'].cpu().tolist(),'raw':raw,'validation':validation,'error':error,'revision':vlm['revision']}
   rows.append(row);(O/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');print(variant,case['id'],limit,raw,flush=True)

if __name__=='__main__':main()
