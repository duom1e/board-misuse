"""Targeted developer probe of crop identity; no detection suppression."""
import json,hashlib,math
from pathlib import Path
from PIL import Image,ImageDraw
from region_vlm_runtime import LocalRegionModel
B=Path(__file__).resolve().parent;E=B/'detector_candidate_vlm_v1'
PROMPT='Identify the main central object in this image crop. Return exactly one label: cutting_board, food_container, table_surface, other, or unknown. cutting_board means a distinct board used for cutting, not a plate or a kitchen worktop. Food and hands may cover it. Use unknown when the visible boundary or object identity is insufficient. Do not infer hygiene, contact, raw/cooked state or compliance.'
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 E.mkdir(exist_ok=True);assert not (E/'locked_plan.json').exists(),'Do not restart'
 source=B/'board_cohort_v1';cases=json.loads((source/'locked_plan.json').read_text(encoding='utf-8'))['cases'];rows=json.loads((source/'results.json').read_text(encoding='utf-8'))['rows'];selected=[];seen=set()
 for positive in [True,False]:
  n=0
  for r in rows:
   if positive and (r['model']!='default' or not r['matched']):continue
   if not positive and not r['negative_false_frame']:continue
   if r['case'] in seen:continue
   predictions=[q for q in r['predictions'] if (q['iou']>=.5 if positive else True)]
   if not predictions:continue
   q=max(predictions,key=lambda q:q['confidence']);c=cases[r['case']];selected.append(dict(image=c['image'],image_sha256=c['sha256'],box=q['box'],detector=r['model'],expected='cutting_board' if positive else 'not_cutting_board',source_group=r['group']));seen.add(r['case']);n+=1
   if n==6:break
 save('locked_plan.json',dict(cases=selected,prompt=PROMPT,margin=.15,max_new_tokens=12,scope='Targeted already-consumed developer selection using known outcomes. No new blind accuracy, training or automatic suppression. Matched positives vs verified no-visible-board negatives only.'))
 model=LocalRegionModel();output=[]
 for i,c in enumerate(selected):
  p=Path(c['image']);assert hashlib.sha256(p.read_bytes()).hexdigest()==c['image_sha256'];im=Image.open(p).convert('RGB');x,y,z,k=c['box'];w=z-x;h=k-y;box=[max(0,math.floor(x-.15*w)),max(0,math.floor(y-.15*h)),min(im.width,math.ceil(z+.15*w)),min(im.height,math.ceil(k+.15*h))];crop=im.crop(box);crop.save(E/f'{i:02}_crop.jpg');raw=model.predict(crop,prompt=PROMPT,max_new_tokens=12);category=raw.strip().lower();category=category if category in {'cutting_board','food_container','table_surface','other','unknown'} else 'unknown';output.append(dict(**c,crop_box=box,raw_response=raw,opinion=category,status='needs_review',detection_removed=False,compliance_determined=False));save('partial_results.json',dict(rows=output));print(i,c['expected'],category,flush=True)
 save('results.json',dict(rows=output,training_performed=False,adopted=False))
if __name__=='__main__':main()
