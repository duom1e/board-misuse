"""Rebuild all selected old FoodSeg labels from masks before deduplicated merge."""
import json,io,hashlib,shutil
from pathlib import Path
import cv2,numpy as np,pyarrow.parquet as pq
from PIL import Image
from prepare_food_taxonomy_alignment import MEAT,VEG
B=Path(__file__).resolve().parent;E=B/'food_aligned_replay_v1';RAW=B.parent/'work/datasets/foodseg103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);base=B/'food_multiveg960_v1';plan=json.loads((base/'locked_plan.json').read_text(encoding='utf-8'));oldpaths=(base/'train.txt').read_text(encoding='utf-8').splitlines();records=[];hashes={};fs={}
 for index,src in enumerate(oldpaths):
  p=Path(src);im=E/'images/train'/f'old_{index}.jpg';lp=E/'labels/train'/f'old_{index}.txt';im.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,im);source_label=p.parent.parent.parent/'labels'/p.parent.name/(p.stem+'.txt');shutil.copyfile(source_label,lp)
  r=dict(image=str(im),label=str(lp),source=src,sha256=sha(im),label_rebuilt=False);records.append(r);hashes[r['sha256']]=r
  if p.stem.startswith('fs') and p.stem[2:].isdigit():fs[int(p.stem[2:])]=r
 found=set()
 for file in sorted(RAW.glob('train-*.parquet')):
  for batch in pq.ParquetFile(file).iter_batches(batch_size=16,columns=['id','label']):
   for raw in batch.to_pylist():
    if raw['id'] not in fs:continue
    r=fs[raw['id']];image=Image.open(r['image']);w,h=image.size;mask=np.array(Image.open(io.BytesIO(raw['label']['bytes'])).resize((w,h),Image.Resampling.NEAREST));lines=[]
    for cls,ids in [(0,MEAT),(1,VEG)]:
     _,_,stats,_=cv2.connectedComponentsWithStats(np.isin(mask,list(ids)).astype('uint8'),8)
     for x,y,bw,bh,area in stats[1:]:
      if area>=.002*w*h and min(bw,bh)>=6:lines.append(f'{cls} {(x+bw/2)/w:.8f} {(y+bh/2)/h:.8f} {bw/w:.8f} {bh/h:.8f}')
    old=Path(r['label']).read_text(encoding='utf-8');new='\n'.join(lines)+('\n' if lines else '');r['label_changed']=old!=new;r['label_rebuilt']=True;Path(r['label']).write_text(new,encoding='utf-8');found.add(raw['id'])
 assert found==set(fs)
 extra=json.loads((B/'food_taxonomy_alignment_v1/manifest.json').read_text(encoding='utf-8'))['records'];duplicates=0
 for r in extra:
  if r['official_split']!='train':continue
  if r['sha256'] in hashes:duplicates+=1;continue
  im=E/'images/train'/f'expanded_{r["official_id"]}.jpg';lp=E/'labels/train'/f'expanded_{r["official_id"]}.txt';shutil.copyfile(r['image'],im);shutil.copyfile(r['label'],lp);row=dict(image=str(im),label=str(lp),source=r['image'],sha256=sha(im),label_rebuilt=True);records.append(row);hashes[row['sha256']]=row
 assert len(records)==len(hashes)
 val=[r for r in extra if r['official_split']=='validation'];assert not set(hashes)&{r['sha256'] for r in val}
 for r in records:r['label_sha256']=sha(r['label'])
 for name,ps in [('train',[r['image'] for r in records]),('oldval',plan['old_validation']),('alignedval',[r['image'] for r in val])]: (E/f'{name}.txt').write_text('\n'.join(ps)+'\n',encoding='utf-8')
 for name,validation in [('data','oldval'),('aligned','alignedval')]: (E/f'{name}.yaml').write_text(f'path: {E.as_posix()}\ntrain: train.txt\nval: {validation}.txt\nnames:\n  0: meat\n  1: vegetable\n',encoding='utf-8')
 (E/'manifest.json').write_text(json.dumps(dict(records=records,reconstructed_old_mask_images=len(found),deduplicated_extra_images=duplicates,aligned_validation=val,old_validation_scope='Historical narrow labels kept only for regression; not expanded-taxonomy accuracy.',training_performed=False),ensure_ascii=False,indent=2),encoding='utf-8');print(len(records),'unique training; rebuilt',len(found),'old masks; deduplicated',duplicates)
if __name__=='__main__':main()
