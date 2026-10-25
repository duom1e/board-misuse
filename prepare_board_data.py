"""Build a one-class Open Images baseline; requires downloaded official CSVs."""
import csv
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
RAW = BASE.parent / 'work' / 'datasets'
OUT = BASE / 'board_dataset'
MID = '/m/02pdsw'

def download(item):
    split, image_id, rows = item
    dest = OUT / 'images' / split / f'{image_id}.jpg'
    url = f'https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg'
    for attempt in range(3):
        try:
            if not dest.exists():
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                dest.write_bytes(response.content)
            with Image.open(dest) as im:
                im.verify()
            labels = []
            for r in rows:
                x1,x2,y1,y2 = [float(r[k]) for k in ('XMin','XMax','YMin','YMax')]
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    raise ValueError('Invalid box')
                labels.append(f'0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}')
            (OUT / 'labels' / split / f'{image_id}.txt').write_text('\n'.join(labels)+'\n')
            return {'split':split, 'image_id':image_id, 'url':url, 'boxes':len(labels),
                    'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)

def main():
    items = []
    for split in ('train','validation','test'):
        for kind in ('images','labels'):
            (OUT / kind / split).mkdir(parents=True, exist_ok=True)
        grouped = defaultdict(list)
        with (RAW / f'oi_{split}.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                if r['LabelName'] == MID and r['IsGroupOf']=='0' and r['IsDepiction']=='0':
                    grouped[r['ImageID']].append(r)
        items += [(split,k,v) for k,v in sorted(grouped.items())]
        print(split, len(grouped), 'images', sum(map(len,grouped.values())), 'boxes',flush=True)
    manifest=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download,item) for item in items]
        for i,f in enumerate(as_completed(futures),1):
            manifest.append(f.result())
            if i%25==0 or i==len(items):
                print('Downloaded',i,'/',len(items),flush=True)
    # Keep test/validation in preference to train if exact images repeat across splits.
    seen={}
    kept=[]
    excluded=[]
    for r in sorted(manifest,key=lambda r:({'test':0,'validation':1,'train':2}[r['split']],r['image_id'])):
        if r['sha256'] in seen:
            excluded.append(r)
            continue
        seen[r['sha256']]=r
        kept.append(r)
    for split in ('train','validation','test'):
        paths=[(OUT/'images'/split/f"{r['image_id']}.jpg").as_posix() for r in kept if r['split']==split]
        (OUT/f'{split}.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
    (OUT/'data.yaml').write_text(f'path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n  0: cutting_board\n',encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({'source':'Open Images official bounding boxes','class_mid':MID,
        'images':kept,'excluded_exact_duplicates':excluded},indent=2),encoding='utf-8')
    preview=Image.new('RGB',(960,720),'white')
    selected=[r for r in kept if r['split']=='train'][:12]
    for i,r in enumerate(selected):
        p=OUT/'images'/r['split']/f"{r['image_id']}.jpg"
        with Image.open(p) as image:
            im=image.convert('RGB')
        draw=ImageDraw.Draw(im)
        for line in (OUT/'labels'/r['split']/f"{r['image_id']}.txt").read_text().splitlines():
            _,x,y,w,h=map(float,line.split()); iw,ih=im.size
            draw.rectangle(((x-w/2)*iw,(y-h/2)*ih,(x+w/2)*iw,(y+h/2)*ih),outline='red',width=4)
        im.thumbnail((320,220))
        px=(i%3)*320; py=(i//3)*180
        im.thumbnail((320,160))
        preview.paste(im,(px,py))
        ImageDraw.Draw(preview).text((px+4,py+161),r['image_id'],fill='black')
    preview.save(BASE/'data_preview.jpg')
    print('FINAL',json.dumps({s:sum(r['split']==s for r in kept) for s in ('train','validation','test')}),flush=True)

if __name__ == '__main__':
    main()
