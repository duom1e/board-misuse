"""Expand official train masks only; preserve clean local supplements and held validation."""
import io,json,hashlib
from pathlib import Path
import numpy as np,cv2,pyarrow.parquet as pq
from PIL import Image
from prepare_food_taxonomy_alignment import MEAT,VEG as BASE_VEG
VEG=BASE_VEG|{59}
B=Path(__file__).resolve().parent;E=B/'food_fullmask_corn_replay_v1';RAW=B.parent/'work/datasets/foodseg103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);records=[];seen={};paths=[]
 held=(B/'food_empty_label_clean_training_v1/val.txt').read_text().splitlines()+(B/'food_aligned_replay_v1/alignedval.txt').read_text().splitlines();heldsha=set(map(sha,held));scanned=0
 for file in sorted(RAW.glob('train-*.parquet')):
  for batch in pq.ParquetFile(file).iter_batches(batch_size=16,columns=['image','label','classes_on_image','id']):
   for r in batch.to_pylist():
    scanned+=1
    if not (MEAT|VEG)&set(r['classes_on_image'] or []):continue
    im=Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB');w,h=im.size;mask=np.array(Image.open(io.BytesIO(r['label']['bytes'])).resize(im.size,Image.Resampling.NEAREST));boxes=[]
    for cls,ids in [(0,MEAT),(1,VEG)]:
     _,_,stats,_=cv2.connectedComponentsWithStats(np.isin(mask,list(ids)).astype('uint8'),8)
     for x,y,bw,bh,area in stats[1:]:
      if area>=.002*w*h and min(bw,bh)>=6:boxes.append((cls,int(x),int(y),int(bw),int(bh)))
    if not boxes:continue
    buf=io.BytesIO();im.save(buf,format='JPEG',quality=92);payload=buf.getvalue();digest=hashlib.sha256(payload).hexdigest()
    if digest in seen or digest in heldsha:continue
    ip=E/f'images/train/fs{r["id"]:06d}.jpg';lp=E/f'labels/train/fs{r["id"]:06d}.txt';ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True);ip.write_bytes(payload);lp.write_text(''.join(f'{c} {(x+bw/2)/w:.8f} {(y+bh/2)/h:.8f} {bw/w:.8f} {bh/h:.8f}\n' for c,x,y,bw,bh in boxes));seen[digest]=str(ip);paths.append(str(ip));records.append(dict(image=str(ip),label=str(lp),sha256=digest,official_id=r['id'],official_split='train',boxes=boxes))
  print('processed',file.name,'selected',len(records),flush=True)
 base=(B/'empty_label_clean_replay_v1/train.txt').read_text().splitlines();supplement=[]
 for p in base:
  digest=sha(p)
  if digest in seen:continue
  assert digest not in heldsha;supplement.append(p)
 paths+=supplement
 (E/'train.txt').write_text('\n'.join(paths)+'\n');(E/'manifest.json').write_text(json.dumps(dict(records=records,local_supplement_entries=supplement,official_scanned=scanned,class_ids=dict(meat=sorted(MEAT),vegetable=sorted(VEG)),train_entries=len(paths),unique_images=len(set(map(sha,paths))),scope='Official training masks only, broad meat/plant taxonomy. Existing clean local supplements retained including repeated exposures. Held hashes excluded. No claim of independent-source generalization; mask boxes are coarse and tiny objects filtered.'),indent=2),encoding='utf-8');print('READY',len(paths),len(records),len(supplement),flush=True)
if __name__=='__main__':main()
