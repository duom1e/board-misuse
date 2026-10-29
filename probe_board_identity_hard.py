import json,hashlib,math
from pathlib import Path
from PIL import Image,ImageDraw
from region_vlm_runtime import LocalRegionModel
from review_detector_candidates import PROMPT
B=Path(__file__).resolve().parent;E=B/'board_identity_hard_v1'
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 E.mkdir(exist_ok=True);assert not (E/'locked_plan.json').exists(),'Do not restart'
 records=json.loads((B/'board_cohort_v1/locked_plan.json').read_text(encoding='utf-8'))['cases'];cases=[x for x in records if x['stratum'] in ('small_edge_partial','hand_occluded_round_board')];assert len(cases)==8
 save('locked_plan.json',dict(cases=cases,prompt=PROMPT,margin=.15,thumbnail=512,max_new_tokens=12,scope='Eight known hard manual board ROIs from consumed single video; manual boxes bypass detector. No independent accuracy, detector recall recovery, training or suppression.'))
 model=LocalRegionModel();rows=[];sheet=Image.new('RGB',(1280,1000));draw=ImageDraw.Draw(sheet)
 for i,c in enumerate(cases):
  p=Path(c['image']);assert hashlib.sha256(p.read_bytes()).hexdigest()==c['sha256'];im=Image.open(p).convert('RGB');x,y,z,k=c['box'];w=z-x;h=k-y;box=[max(0,math.floor(x-.15*w)),max(0,math.floor(y-.15*h)),min(im.width,math.ceil(z+.15*w)),min(im.height,math.ceil(k+.15*h))];crop=im.crop(box);crop.save(E/f'{i:02}_crop.jpg');raw=model.predict(crop,prompt=PROMPT,max_new_tokens=12);opinion=raw.strip().lower();opinion=opinion if opinion in {'cutting_board','food_container','table_surface','other','unknown'} else 'unknown';rows.append(dict(**c,crop_box=box,raw_response=raw,opinion=opinion,expected='cutting_board',detection_removed=False,status='needs_review'));save('partial_results.json',dict(rows=rows));crop.thumbnail((310,420));a=i%4*320;b=i//4*500;sheet.paste(crop,(a,b));draw.text((a,b+430),f'{i} {c["stratum"]}\n{opinion}',fill='white');print(i,opinion,flush=True)
 sheet.save(E/'review.jpg');save('results.json',dict(rows=rows,training_performed=False,adopted=False))
if __name__=='__main__':main()
