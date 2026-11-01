"""Rebuild existing development validation masks with same broad taxonomy plus corn."""
import io,json,shutil
from pathlib import Path
import cv2,numpy as np,pyarrow.parquet as pq
from PIL import Image
from prepare_food_taxonomy_alignment import MEAT,VEG as BASE_VEG
VEG=BASE_VEG|{59};B=Path(__file__).resolve().parent;E=B/'food_corn_validation_v1'
def main():
 E.mkdir(exist_ok=False);wanted={};groups={'old64':[],'fishfruit30':[]}
 for p in (B/'food_empty_label_clean_training_v1/val.txt').read_text().splitlines():
  key=int(Path(p).stem[2:]);wanted[key]=p;groups['old64'].append(key)
 for r in json.loads((B/'food_taxonomy_alignment_v1/manifest.json').read_text())['records']:
  if r['official_split']=='validation':wanted[r['official_id']]=r['image'];groups['fishfruit30'].append(r['official_id'])
 done={};changes=[]
 for file in sorted((B.parent/'work/datasets/foodseg103').glob('validation-*.parquet')):
  for batch in pq.ParquetFile(file).iter_batches(batch_size=16,columns=['id','label','classes_on_image']):
   for r in batch.to_pylist():
    if r['id'] not in wanted:continue
    src=Path(wanted[r['id']]);w,h=Image.open(src).size;arr=np.array(Image.open(io.BytesIO(r['label']['bytes'])).resize((w,h),Image.Resampling.NEAREST));boxes=[]
    for c,ids in [(0,MEAT),(1,VEG)]:
     _,_,stats,_=cv2.connectedComponentsWithStats(np.isin(arr,list(ids)).astype('uint8'),8)
     for x,y,bw,bh,area in stats[1:]:
      if area>=.002*w*h and min(bw,bh)>=6:boxes.append((c,int(x),int(y),int(bw),int(bh)))
    ip=E/f'images/validation/fs{r["id"]:06d}.jpg';lp=E/f'labels/validation/fs{r["id"]:06d}.txt';ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,ip);lp.write_text(''.join(f'{c} {(x+bw/2)/w:.8f} {(y+bh/2)/h:.8f} {bw/w:.8f} {bh/h:.8f}\n' for c,x,y,bw,bh in boxes));done[r['id']]=str(ip);changes.append(dict(official_id=r['id'],source=str(src),image=str(ip),boxes=boxes,has_corn=59 in r['classes_on_image']))
 assert set(done)==set(wanted)
 for name,ids in groups.items():
  (E/f'{name}.txt').write_text('\n'.join(done[i] for i in ids)+'\n');(E/f'{name}.yaml').write_text(f'path: {E.as_posix()}\ntrain: {name}.txt\nval: {name}.txt\nnames:\n  0: meat\n  1: vegetable\n')
 (E/'manifest.json').write_text(json.dumps(dict(records=changes,groups=groups,scope='Existing development images, new broad labels plus corn. Not a new independent validation cohort. YAML train key is compatibility only; never train these images.'),indent=2));print('validation unique',len(done),'corn',sum(r['has_corn'] for r in changes))
if __name__=='__main__':main()
