"""Strict YOLO detection-data audit over the actual split lists, not whole folders."""
import hashlib
import json
import math
import os
from collections import Counter,defaultdict
from pathlib import Path

import yaml
from PIL import Image
from check_split_leakage import dhash

EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}

def absolute(path,base):
    path=Path(path)
    return Path(os.path.abspath(path if path.is_absolute() else Path(base)/path))

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def image_label_path(image):
    parts=list(Path(image).parts)
    indexes=[i for i,p in enumerate(parts) if p.lower()=='images']
    if not indexes:raise ValueError(f'Image path must contain an images directory: {image}')
    parts[indexes[-1]]='labels'
    return Path(*parts).with_suffix('.txt')

def split_images(value,root):
    values=value if isinstance(value,list) else [value]
    files=[]
    for entry in values:
        if not isinstance(entry,str):raise ValueError('Split entries must be paths')
        path=absolute(entry,root)
        if path.is_dir():
            files.extend(sorted((p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS),key=str))
        elif path.is_file() and path.suffix.lower()=='.txt':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():files.append(absolute(line.strip(),path.parent))
        elif path.is_file() and path.suffix.lower() in EXTENSIONS:files.append(path)
        else:raise FileNotFoundError(f'Split path not found or unsupported: {path}')
    if not files:raise ValueError('Every split must contain images')
    if len(files)!=len(set(map(str,files))):raise ValueError('Repeated image path within a split list')
    return files

def read_boxes(label,names):
    if not label.is_file():raise ValueError(f'Missing label file (not assumed negative): {label}')
    boxes=[]
    for line_no,line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():continue
        tokens=line.split()
        if len(tokens)!=5:raise ValueError(f'Expected 5 detection-label fields: {label}:{line_no}')
        values=list(map(float,tokens));cls,x,y,w,h=values
        if not all(math.isfinite(v) for v in values) or not cls.is_integer() or not 0<=cls<len(names):
            raise ValueError(f'Invalid class or nonfinite number: {label}:{line_no}')
        if w<=0 or h<=0 or x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2>1+1e-6 or y+h/2>1+1e-6:
            raise ValueError(f'Invalid normalized box: {label}:{line_no}')
        boxes.append([int(cls),x,y,w,h])
    return boxes

def audit_dataset(data_file,near_threshold=4,groups_file=None,review_file=None):
    if not isinstance(near_threshold,int) or not 0<=near_threshold<=64:raise ValueError('Invalid dHash threshold')
    data_file=Path(data_file).resolve()
    data=yaml.safe_load(data_file.read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict):raise ValueError('Dataset YAML must be a mapping')
    names=data.get('names')
    if isinstance(names,dict):
        if sorted(names)!=list(range(len(names))):raise ValueError('Class IDs must be contiguous integers starting at zero')
        names=[names[i] for i in range(len(names))]
    if not isinstance(names,list) or not names or not all(isinstance(n,str) and n.strip() for n in names) or len(names)!=len(set(names)):
        raise ValueError('Expected nonempty unique class names')
    root=absolute(data.get('path','.'),data_file.parent)
    splits={s:split_images(data.get(s),root) for s in ['train','val','test']}
    groups={}
    if groups_file:
        raw=json.loads(Path(groups_file).read_text(encoding='utf-8-sig'))
        groups={str(absolute(p,root).resolve()):g for p,g in raw.items()}
        if not all(isinstance(g,str) and g.strip() for g in groups.values()):raise ValueError('Group IDs must be nonempty strings')
    approved=set()
    if review_file:
        for row in json.loads(Path(review_file).read_text(encoding='utf-8-sig')):
            if row.get('verdict')=='different_images' and row.get('reason') and row.get('reviewer'):
                approved.add(tuple(sorted([row['sha256_a'],row['sha256_b']])))
    records=[];errors=[];counts={}
    for split,images in splits.items():
        classes=Counter();backgrounds=0
        for image in images:
            try:
                if not image.is_file() or image.suffix.lower() not in EXTENSIONS:raise ValueError(f'Invalid image: {image}')
                label=image_label_path(image)
                boxes=read_boxes(label,names)
                with Image.open(image) as im:
                    width,height=im.size;im.verify()
                with Image.open(image) as im:
                    rgb=im.convert('RGB')
                    pixel_digest=hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode()+rgb.tobytes()).hexdigest()
                group=groups.get(str(image.resolve()))
                if groups_file and group is None:raise ValueError(f'Group manifest missing image: {image}')
                row={'split':split,'image':str(image),'label':str(label),'width':width,'height':height,
                    'image_sha256':sha256(image),'decoded_rgb_sha256':pixel_digest,'label_sha256':sha256(label),'dhash':dhash(image),'group':group,
                    'boxes':len(boxes),'class_counts':dict(Counter(names[b[0]] for b in boxes))}
                records.append(row);classes.update(row['class_counts']);backgrounds+=not boxes
            except Exception as error:errors.append({'split':split,'image':str(image),'error':f'{type(error).__name__}: {error}'})
        counts[split]={'images':len(images),'valid_images':sum(r['split']==split for r in records),
            'boxes':sum(r['boxes'] for r in records if r['split']==split),'explicit_empty_labels':backgrounds,'class_boxes':dict(classes)}
    by_hash=defaultdict(list)
    for row in records:by_hash[row['image_sha256']].append(row)
    duplicates=[];within=[]
    for digest,rows in by_hash.items():
        if len(rows)>1:
            target=duplicates if len({r['split'] for r in rows})>1 else within
            target.append({'sha256':digest,'files':[r['image'] for r in rows],'splits':sorted({r['split'] for r in rows})})
    by_pixels=defaultdict(list)
    for row in records:by_pixels[row['decoded_rgb_sha256']].append(row)
    for digest,rows in by_pixels.items():
        if len(rows)>1 and len({r['image_sha256'] for r in rows})>1:
            target=duplicates if len({r['split'] for r in rows})>1 else within
            target.append({'decoded_rgb_sha256':digest,'type':'identical_decoded_pixels_different_file_bytes',
                'files':[r['image'] for r in rows],'splits':sorted({r['split'] for r in rows})})
    near=[]
    for i,left in enumerate(records):
        for right in records[i+1:]:
            if left['split']==right['split'] or left['decoded_rgb_sha256']==right['decoded_rgb_sha256']:continue
            distance=(left['dhash']^right['dhash']).bit_count()
            if distance<=near_threshold:
                pair=tuple(sorted([left['image_sha256'],right['image_sha256']]))
                near.append({'a':left['image'],'b':right['image'],'sha256_a':pair[0],'sha256_b':pair[1],
                    'distance':distance,'reviewed_as_different':pair in approved})
    by_group=defaultdict(list)
    for row in records:
        if row['group']:by_group[row['group']].append(row)
    shared_groups=[{'group':key,'splits':sorted({r['split'] for r in rows})} for key,rows in by_group.items() if len({r['split'] for r in rows})>1]
    warnings=[]
    if not groups_file:warnings.append('No source-scene/video group manifest supplied; scene independence is not verified.')
    if any(c['explicit_empty_labels'] for c in counts.values()):warnings.append('Empty label files are explicit declarations, not proof that no target is visible. Review negative samples visually.')
    if within:warnings.append('Exact duplicates within a split can bias sampling and metrics.')
    class_names=set(names)
    for split,count in counts.items():
        missing=sorted(class_names-set(count['class_boxes']))
        if missing:errors.append({'split':split,'error':f'Classes have no labeled boxes: {missing}'})
    fingerprint=hashlib.sha256(json.dumps({'names':names,'records':[{k:r[k] for k in ['split','image','image_sha256','label_sha256']} for r in records]},sort_keys=True).encode()).hexdigest()
    passed=not errors and not duplicates and not shared_groups and not any(not r['reviewed_as_different'] for r in near)
    return {'passed':passed,'data_file':str(data_file),'names':names,'counts':counts,'errors':errors,
        'cross_split_exact_duplicates':duplicates,'within_split_exact_duplicates':within,
        'cross_split_near_duplicates':near,'cross_split_groups':shared_groups,'near_threshold':near_threshold,
        'warnings':warnings,'records':records,'dataset_fingerprint':fingerprint,
        'limitation':'A clean hash/dHash screen cannot establish clean annotations or independent scenes.'}

def freeze_dataset(audit,destination):
    destination=Path(destination);destination.mkdir(parents=True)
    for split in ['train','val','test']:
        (destination/f'{split}.txt').write_text('\n'.join(r['image'] for r in audit['records'] if r['split']==split)+'\n',encoding='utf-8')
    config={'path':str(destination),'train':'train.txt','val':'val.txt','test':'test.txt','names':audit['names']}
    target=destination/'data.yaml';target.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    return target

def verify_snapshot(audit):
    for row in audit['records']:
        if sha256(row['image'])!=row['image_sha256'] or sha256(row['label'])!=row['label_sha256']:
            raise RuntimeError(f"Dataset changed after plan lock: {row['image']}")
