"""Reject quarantined images, unreviewed empty labels and held-set leakage before training."""
import argparse,json,hashlib,math
from pathlib import Path
B=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(train_list,extra_reviewed=None):
 removed=json.loads((B/'empty_label_clean_replay_v1/manifest.json').read_text())['removed_records'];blocked={digest(r['image']) for r in removed}
 reviewed=json.loads((B/'empty_label_visual_audit_v1/manifest.json').read_text());allowed_empty={digest(r['image']) for r in reviewed}-blocked
 if extra_reviewed:
  for r in json.loads(Path(extra_reviewed).read_text())['records']:
   assert r['review_status']=='visually_checked_no_target_nonexpert' and digest(r['image'])==r['sha256']
   allowed_empty.add(r['sha256'])
 held=(B/'food_empty_label_clean_training_v1/val.txt').read_text().splitlines()+(B/'food_aligned_replay_v1/alignedval.txt').read_text().splitlines()
 for name in ['kitchen_source_validation_v1','food_group_boxes_v1']:
  for r in json.loads((B/name/'manifest.json').read_text(encoding='utf-8'))['records']:
   if name=='kitchen_source_validation_v1' or r.get('split')=='source_held_development':held.append(r['image'])
 held_hash={digest(p) for p in held};errors=[];seen={};empty=0
 paths=Path(train_list).read_text().splitlines()
 for p in sorted(set(paths)):
  ip=Path(p)
  if not ip.exists():errors.append(dict(image=p,reason='missing_image'));continue
  sha=digest(ip);seen[p]=sha
  if sha in blocked:errors.append(dict(image=p,reason='quarantined_image'))
  if sha in held_hash:errors.append(dict(image=p,reason='held_image_leakage'))
  lp=Path(str(ip).replace('\\images\\','\\labels\\').replace('/images/','/labels/')).with_suffix('.txt')
  if not lp.exists():errors.append(dict(image=p,reason='missing_label'));continue
  lines=lp.read_text().splitlines()
  if not lines:
   empty+=1
   if sha not in allowed_empty:errors.append(dict(image=p,reason='empty_label_not_visually_reviewed'))
  for line in lines:
   try:
    vals=list(map(float,line.split()));assert len(vals)==5 and all(math.isfinite(x) for x in vals) and vals[0] in [0,1] and all(0<=x<=1 for x in vals[1:]) and vals[3]>0 and vals[4]>0
   except (ValueError,AssertionError):errors.append(dict(image=p,reason='invalid_yolo_label'))
 return dict(ok=not errors,entries=len(paths),unique_paths=len(set(paths)),unique_image_hashes=len(set(seen.values())),empty_images=empty,errors=errors,scope='Exact-byte identity and reviewed-empty gate; no near-duplicate or complete semantic validation.')
def main():
 p=argparse.ArgumentParser();p.add_argument('train_list');p.add_argument('--output',required=True);a=p.parse_args();result=audit(a.train_list);Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='errors'});raise SystemExit(0 if result['ok'] else 1)
if __name__=='__main__':main()
