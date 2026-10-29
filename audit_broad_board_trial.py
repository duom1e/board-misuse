import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'broad_board_trial_v1'
def main():
 r=json.loads((E/'results.json').read_text(encoding='utf-8'));plan=json.loads((E/'locked_plan.json').read_text(encoding='utf-8'));assert hashlib.sha256((B/'active_models.json').read_bytes()).hexdigest()==plan['active_models_sha256'];assert len((E/'train/results.csv').read_text().splitlines())==11
 train_hashes={hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (E/'train.txt').read_text(encoding='utf-8').splitlines() if p.strip()}
 assert all(hashlib.sha256(Path(x['image']).read_bytes()).hexdigest() not in train_hashes for x in plan['heldout'])
 summary={}
 for name,m in r['models'].items():
  out=dict(regression=m['regression'])
  for group in ['new_train_fit','heldout_consumed_diagnostic']:
   rows=m[group];out[group]=dict(matches=sum(x['matched'] for x in rows),positive_count=sum(bool(x['expected']) for x in rows),negative_count=sum(x['expected'] is None for x in rows),negative_false_frames=sum(bool(x['predictions']) for x in rows if x['expected'] is None),unmatched_boxes=sum(len(x['predictions'])-int(x['matched']) for x in rows))
  summary[name]=out;rows=m['heldout_consumed_diagnostic'];s=Image.new('RGB',(1280,1200))
  for i,x in enumerate(rows):
   im=Image.open(x['image']).convert('RGB');d=ImageDraw.Draw(im);d.rectangle(x['expected'],outline='lime',width=3)
   for q in x['predictions']:d.rectangle(q['box'],outline='orange',width=3)
   im.thumbnail((640,370));a=i%2*640;b=i//2*400;s.paste(im,(a,b));ImageDraw.Draw(s).text((a,b+375),f"{name} {i} match={x['matched']}",fill='white')
  s.save(E/f'{name}_heldout.jpg')
 summary.update(adopted=False,active_models_unchanged=True,train_images=len(train_hashes),heldout_image_hash_overlap=0,training_performed=True,fresh_blind_test=False)
 (E/'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
if __name__=='__main__':main()
