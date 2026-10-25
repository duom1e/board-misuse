"""Single-class vegetable subset, preserving Open Images official splits."""
import csv
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from PIL import Image, ImageDraw

BASE=Path(__file__).resolve().parent
RAW=BASE.parent/'work'/'datasets'
OUT=BASE/'vegetable_dataset'
MID='/m/0f4s2w'

def download(item):
    split,image_id,rows=item
    dest=OUT/'images'/split/f'{image_id}.jpg'
    url=f'https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg'
    for attempt in range(3):
        try:
            if not dest.exists():
                r=requests.get(url,timeout=40);r.raise_for_status();dest.write_bytes(r.content)
            with Image.open(dest) as im: im.verify()
            labels=[]
            for r in rows:
                x1,x2,y1,y2=[float(r[k]) for k in ('XMin','XMax','YMin','YMax')]
                if not (0<=x1<x2<=1 and 0<=y1<y2<=1): raise ValueError('Bad box')
                labels.append(f'0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}')
            (OUT/'labels'/split/f'{image_id}.txt').write_text('\n'.join(labels)+'\n')
            return {'split':split,'image_id':image_id,'boxes':len(labels),'url':url,
                    'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        except Exception:
            if attempt==2: raise

def main():
    items=[]
    available={}
    for split,cap in [('train',500),('validation',60),('test',100)]:
        for kind in ['images','labels']: (OUT/kind/split).mkdir(parents=True,exist_ok=True)
        groups=defaultdict(list)
        with (RAW/f'oi_{split}.csv').open() as f:
            header=next(csv.reader([next(f)]))
            for line in f:
                if ','+MID+',' not in line: continue
                r=dict(zip(header,next(csv.reader([line]))))
                if r['IsGroupOf']=='0' and r['IsDepiction']=='0': groups[r['ImageID']].append(r)
        ids=sorted(groups);available[split]=len(ids);random.Random(42).shuffle(ids)
        items += [(split,i,groups[i]) for i in ids[:cap]]
        print(split,'available',len(ids),'selected',min(cap,len(ids)),flush=True)
    manifest=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(download,item) for item in items]
        for i,f in enumerate(as_completed(fs),1):
            manifest.append(f.result())
            if i%50==0 or i==len(items): print('downloaded',i,'/',len(items),flush=True)
    kept=[];excluded=[];seen=set()
    for r in sorted(manifest,key=lambda r:({'test':0,'validation':1,'train':2}[r['split']],r['image_id'])):
        if r['sha256'] in seen: excluded.append(r);continue
        seen.add(r['sha256']);kept.append(r)
    for split in ['train','validation','test']:
        (OUT/f'{split}.txt').write_text('\n'.join((OUT/'images'/split/f"{r['image_id']}.jpg").as_posix() for r in kept if r['split']==split)+'\n')
    (OUT/'data.yaml').write_text(f'path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n  0: vegetable\n')
    (OUT/'manifest.json').write_text(json.dumps({'source':'Open Images official Vegetable boxes',
        'mid':MID,'sampling_seed':42,'available_images':available,'images':kept,'excluded_exact_duplicates':excluded},indent=2))
    selected_ids={r['image_id'] for r in kept};attribution=[]
    for split in ['train','validation','test']:
        with (RAW/f'oi_meta_{split}.csv').open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                if r['ImageID'] in selected_ids: r['DatasetSplit']=split;attribution.append(r)
    if {r['ImageID'] for r in attribution}!=selected_ids: raise RuntimeError('Missing attribution')
    with (OUT/'attribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(attribution[0]));w.writeheader();w.writerows(attribution)
    preview=Image.new('RGB',(960,720),'white')
    for i,r in enumerate([r for r in kept if r['split']=='train'][:12]):
        with Image.open(OUT/'images'/'train'/f"{r['image_id']}.jpg") as im: image=im.convert('RGB')
        draw=ImageDraw.Draw(image);iw,ih=image.size
        for line in (OUT/'labels'/'train'/f"{r['image_id']}.txt").read_text().splitlines():
            _,x,y,w,h=map(float,line.split());draw.rectangle(((x-w/2)*iw,(y-h/2)*ih,(x+w/2)*iw,(y+h/2)*ih),outline='red',width=3)
        image.thumbnail((320,160));px=(i%3)*320;py=(i//3)*180;preview.paste(image,(px,py));ImageDraw.Draw(preview).text((px+4,py+161),r['image_id'],fill='black')
    preview.save(BASE/'vegetable_data_preview.jpg')
    print('READY',json.dumps({s:sum(r['split']==s for r in kept) for s in ['train','validation','test']}),flush=True)

if __name__=='__main__': main()
