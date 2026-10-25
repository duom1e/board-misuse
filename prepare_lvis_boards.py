"""New LVIS chopping-board data, CC BY 2.0 image metadata only.

Preserves explicit negative-category labels and excludes previously seen photo IDs/content.
Does not run either detector on the fresh test images.
"""
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.parse import urlparse
import requests
from PIL import Image,ImageDraw
import yaml

from experiment_data import dhash,sha256

BASE=Path(__file__).resolve().parent
RAW=BASE.parent/'work'/'datasets'/'lvis'
OUT=BASE/'board_lvis_ccby'

def flickr_id(url):
    match=re.search(r'/photos/[^/]+/(\d+)',url)
    if not match:match=re.search(r'/(\d+)_[^/]+\.(?:jpe?g|png)',url,re.I)
    return match.group(1) if match else None

def prior_photo_ids():
    used=set()
    for file in BASE.rglob('attribution.csv'):
        if OUT in file.parents:continue
        try:
            for r in csv.DictReader(file.open(encoding='utf-8-sig')):
                for key in ['OriginalURL','OriginalLandingURL','flickr_url']:
                    identifier=flickr_id(r.get(key,'') or '')
                    if identifier:used.add(identifier)
        except (UnicodeError,csv.Error):continue
    return used

def prior_content_index():
    cache=RAW/'prior_image_index.json'
    if cache.exists():return json.loads(cache.read_text(encoding='utf-8'))
    files=[];seen=set()
    for directory in BASE.iterdir():
        if not directory.is_dir() or directory==OUT or directory.name in {'experiments','batch_results','misuse_results','predictions','training_runs','__pycache__'}:continue
        image_root=directory/'images'
        if not image_root.exists():continue
        for file in image_root.rglob('*'):
            if file.suffix.lower() not in {'.jpg','.jpeg','.png'} or not file.is_file():continue
            stat=file.stat();identity=(stat.st_dev,stat.st_ino,stat.st_size)
            if identity not in seen:seen.add(identity);files.append(file)
    indexed=[];hashes=set()
    for i,file in enumerate(files,1):
        digest=sha256(file)
        if digest in hashes:continue
        hashes.add(digest)
        try:indexed.append({'file':str(file),'sha256':digest,'dhash':dhash(file)})
        except Exception:continue
        if i%1000==0:print('prior_files_indexed',i,'unique',len(indexed),flush=True)
    cache.write_text(json.dumps(indexed),encoding='utf-8')
    return indexed

def fetch(record):
    image=record['image'];relative=urlparse(image['coco_url']).path.lstrip('/')
    url='https://s3.amazonaws.com/images.cocodataset.org/'+relative
    dest=RAW/'images'/f"{image['id']:012d}.jpg";dest.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            if not dest.exists():
                response=requests.get(url,timeout=35);response.raise_for_status();dest.write_bytes(response.content)
            with Image.open(dest) as im:
                if im.size!=(image['width'],image['height']):raise ValueError('Image dimensions disagree with annotation')
                im.verify()
            return {**record,'source_file':str(dest),'download_url':url,'sha256':sha256(dest),'dhash':dhash(dest)}
        except Exception as error:
            if attempt==2:return {**record,'error':f'{type(error).__name__}: {error}'}

def link(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    os.link(source,target)

def main():
    if OUT.exists():raise RuntimeError('Output already exists; preserve completed data build')
    candidates=json.loads((RAW/'lvis_board_candidates.json').read_text(encoding='utf-8'))
    used=prior_photo_ids();selected=[];excluded=[]
    for split,part in candidates['splits'].items():
        for r in part['images']:
            record={**r,'official_lvis_split':split,'flickr_photo_id':flickr_id(r['image']['flickr_url'])}
            if record['flickr_photo_id'] in used:excluded.append({'id':r['image']['id'],'reason':'previously used Flickr photo ID'})
            else:selected.append(record)
    print('metadata_filter',len(selected),'eligible_new',len(excluded),'excluded_old_photo_ids',flush=True)
    done=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fetch,r) for r in selected]
        for i,future in enumerate(as_completed(futures),1):
            done.append(future.result())
            if i%50==0 or i==len(futures):print('downloaded',i,'/',len(futures),flush=True)
    index=prior_content_index();exact={r['sha256']:r for r in index}
    keep=[]
    for record in done:
        if record.get('error'):excluded.append({'id':record['image']['id'],'reason':record['error']});continue
        match=exact.get(record['sha256'])
        if match:
            excluded.append({'id':record['image']['id'],'reason':'previous image bytes','previous':match['file']});continue
        matches=[r for r in index if (record['dhash']^r['dhash']).bit_count()<=4]
        if matches:
            excluded.append({'id':record['image']['id'],'reason':'near previous image, conservatively excluded before model evaluation','previous':[r['file'] for r in matches[:3]]});continue
        keep.append(record)
    # Split by official LVIS image set, and stratify only the train portion for validation.
    split_records={s:[] for s in ['train','val','test']}
    for positive in [False,True]:
        rows=sorted([r for r in keep if r['official_lvis_split']=='train' and bool(r['annotations'])==positive],key=lambda r:r['image']['id'])
        random.Random(20261025).shuffle(rows)
        n=max(1,round(len(rows)*.2)) if rows else 0
        split_records['val']+=rows[:n];split_records['train']+=rows[n:]
    split_records['test']=[r for r in keep if r['official_lvis_split']=='val']
    counts={s:{'images':len(rs),'positive':sum(bool(r['annotations']) for r in rs)} for s,rs in split_records.items()}
    if any(c['positive']<3 for c in counts.values()):raise RuntimeError(f'Too few fresh positive images after filtering: {counts}')
    OUT.mkdir();manifest=[]
    # Preserve the current board training set, but never its old validation/test in new evaluation.
    old=BASE/'board_dataset_v2'
    for line in (old/'train.txt').read_text(encoding='utf-8-sig').splitlines():
        source=Path(line);name='oi_'+source.name
        link(source,OUT/'images'/'train'/name)
        label=OUT/'labels'/'train'/Path(name).with_suffix('.txt');label.parent.mkdir(parents=True,exist_ok=True)
        label.write_bytes((old/'labels'/'train'/source.with_suffix('.txt').name).read_bytes())
        manifest.append({'split':'train','image':str(OUT/'images'/'train'/name),'source_file':str(source),'source':'historical_Open_Images_training_only'})
    for split,records in split_records.items():
        for r in records:
            image=r['image'];name=f"lvis_{image['id']:012d}.jpg"
            link(r['source_file'],OUT/'images'/split/name)
            label=OUT/'labels'/split/Path(name).with_suffix('.txt');label.parent.mkdir(parents=True,exist_ok=True)
            lines=[]
            for a in r['annotations']:
                x,y,w,h=a['bbox'];iw,ih=image['width'],image['height']
                x1=max(0,x);y1=max(0,y);x2=min(iw,x+w);y2=min(ih,y+h)
                if x2<=x1 or y2<=y1:raise ValueError('Invalid source box')
                lines.append(f'0 {(x1+x2)/2/iw:.8f} {(y1+y2)/2/ih:.8f} {(x2-x1)/iw:.8f} {(y2-y1)/ih:.8f}')
            if not lines and not r['negative_explicitly_verified']:raise ValueError('Unverified negative candidate')
            label.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
            manifest.append({**r,'split':split,'image_file':str(OUT/'images'/split/name),
                'source':'LVIS_v1','license_url':'https://creativecommons.org/licenses/by/2.0/',
                'original_photo_page':f"https://www.flickr.com/photo.gne?id={r['flickr_photo_id']}",
                'attribution_note':'LVIS metadata supplies a Flickr image URL and license ID but not author name. Original source link retained; no independent rights verification claimed.'})
    for split in split_records:
        paths=sorted((OUT/'images'/split).glob('*.jpg'))
        (OUT/f'{split}.txt').write_text('\n'.join(str(p) for p in paths)+'\n',encoding='utf-8')
    (OUT/'data.yaml').write_text(yaml.safe_dump({'path':str(OUT),'train':'train.txt','val':'val.txt','test':'test.txt','names':['cutting_board']},sort_keys=False),encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({'records':manifest,'new_lvis_counts':counts,'excluded':excluded,
        'selection':'Only image license ID 4 (CC BY 2.0); exclude not-exhaustive board labels. Explicit LVIS board negatives only.',
        'split_policy':'Historical OI training images only in train. New LVIS train 80/20 stratified; official LVIS val is test.',
        'prior_content_index_images':len(index),'seed':20261025,
        'limitations':['Public household/food photos, not canteen surveillance.','Flickr photo IDs and visual similarity do not prove household/scene independence.','License information comes from source metadata; authors not independently resolved.']},ensure_ascii=False,indent=2),encoding='utf-8')
    positive_train=[r for r in manifest if r.get('source')=='LVIS_v1' and r['split']=='train' and r['annotations']][:12]
    sheet=Image.new('RGB',(1200,960),'white')
    for i,r in enumerate(positive_train):
        im=Image.open(r['source_file']).convert('RGB');draw=ImageDraw.Draw(im)
        for a in r['annotations']:
            x,y,w,h=a['bbox'];draw.rectangle((x,y,x+w,y+h),outline='red',width=3)
        im.thumbnail((400,290));x=i%3*400;y=i//3*240;im.thumbnail((400,215));sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+4,y+218),str(r['image']['id']),fill='black')
    sheet.save(OUT/'train_label_preview.jpg')
    print('READY',json.dumps(counts),'historical_train',sum(r['source']=='historical_Open_Images_training_only' for r in manifest),flush=True)

if __name__=='__main__':main()
