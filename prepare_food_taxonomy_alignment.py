"""New fish/fruit supervision from official FoodSeg masks, no pseudo labels."""
import io,json,hashlib
from pathlib import Path
import cv2,numpy as np,pyarrow.parquet as pq
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;RAW=B.parent/'work/datasets/foodseg103';E=B/'food_taxonomy_alignment_v1'
MEAT={46,47,48,49,50,51,54};VEG=set(range(25,46))|set(range(69,97));FRUIT=set(range(25,46))
def main():
 E.mkdir(exist_ok=False);records=[];counts={};seen=set()
 for split,files,cap in [('train',sorted(RAW.glob('train-*.parquet')),60),('validation',sorted(RAW.glob('validation-*.parquet')),15)]:
  counts[split]={'fish':0,'fruit':0};paths=[]
  for file in files:
   for batch in pq.ParquetFile(file).iter_batches(batch_size=16,columns=['image','label','classes_on_image','id']):
    for r in batch.to_pylist():
     ids=set(r['classes_on_image'] or []);tag='fish' if 54 in ids and counts[split]['fish']<cap else 'fruit' if ids&FRUIT and counts[split]['fruit']<cap else None
     if tag is None:continue
     image=Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB');mask=Image.open(io.BytesIO(r['label']['bytes']));mask=mask.resize(image.size,Image.Resampling.NEAREST);arr=np.array(mask);w,h=image.size;boxes=[]
     for cls,allowed in [(0,MEAT),(1,VEG)]:
      binary=np.isin(arr,list(allowed)).astype('uint8');n,_,stats,_=cv2.connectedComponentsWithStats(binary,8)
      for x,y,bw,bh,area in stats[1:]:
       if area>=.002*w*h and min(bw,bh)>=6:boxes.append([cls,int(x),int(y),int(x+bw),int(y+bh)])
     if not boxes:continue
     raw=io.BytesIO();image.save(raw,format='JPEG',quality=92);payload=raw.getvalue();digest=hashlib.sha256(payload).hexdigest()
     if digest in seen:continue
     seen.add(digest);key=f'{split}_{r["id"]}';ip=E/'images'/split/f'{key}.jpg';lp=E/'labels'/split/f'{key}.txt';ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True);ip.write_bytes(payload)
     lp.write_text(''.join(f'{c} {(a+z)/2/w:.8f} {(b+t)/2/h:.8f} {(z-a)/w:.8f} {(t-b)/h:.8f}\n' for c,a,b,z,t in boxes),encoding='utf-8');paths.append(str(ip));counts[split][tag]+=1
     records.append(dict(id=key,official_id=r['id'],official_split=split,selection=tag,image=str(ip),sha256=digest,label=str(lp),label_sha256=hashlib.sha256(lp.read_bytes()).hexdigest(),boxes=boxes,mask_annotation=True))
   if all(v>=cap for v in counts[split].values()):break
  (E/f'{split}.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
 (E/'data.yaml').write_text(f'path: {E.as_posix()}\ntrain: train.txt\nval: validation.txt\nnames:\n  0: meat\n  1: vegetable\n',encoding='utf-8')
 (E/'manifest.json').write_text(json.dumps(dict(records=records,counts=counts,class_ids=dict(meat=sorted(MEAT),vegetable=sorted(VEG)),scope='Fish54 added to animal meat; fruit25..45 to plant ingredient. Official train/validation separated and decoded-JPEG SHA deduplicated. Mask-to-box connected components, min area .002 and side6; small instances omitted. No raw/cooked/contact labels. Background ignores other foods. Existing narrow-label datasets unchanged. Prepared only, not trained; source images may have appeared in historical models with other labels.'),ensure_ascii=False,indent=2),encoding='utf-8')
 selected=[r for split in ['train','validation'] for tag in ['fish','fruit'] for r in [x for x in records if x['official_split']==split and x['selection']==tag][:4]]
 sheet=Image.new('RGB',(1600,1400),'white')
 for i,r in enumerate(selected):
  im=Image.open(r['image']);d=ImageDraw.Draw(im)
  for cls,a,b,c,t in r['boxes']:d.rectangle([a,b,c,t],outline='red' if cls==0 else 'lime',width=3)
  im.thumbnail((390,300));x=i%4*400;y=i//4*350;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x,y+305),r['id']+' '+r['selection'],fill='black')
 sheet.save(E/'review.jpg');print(counts,len(records))
if __name__=='__main__':main()
