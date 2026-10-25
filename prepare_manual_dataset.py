"""Audit manual exports and build datasets only from explicit, frozen group assignments."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import yaml
from PIL import Image
from annotation_io import sha
from experiment_data import audit_dataset, read_boxes
from check_split_leakage import dhash

BASE = Path(__file__).resolve().parent


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def pool_audit(pool):
    pool = Path(pool).resolve()
    records, errors, incomplete = [], [], []
    for folder in sorted(pool.iterdir()) if pool.exists() else []:
        if not folder.is_dir(): continue
        if not (folder/'COMPLETE').is_file():
            incomplete.append(str(folder)); continue
        try:
            meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8-sig'))
            image, label = folder/'image.png', folder/'image.txt'
            digest = sha(image)
            if digest != meta['exported_image_sha256'] or digest != meta['normalized_sha256']:
                raise ValueError('Image digest differs from recorded normalized/exported image')
            if meta.get('classes') != ['cutting_board'] or meta.get('data_status') != 'unassigned_development':
                raise ValueError('Unsupported classes or provenance status')
            group = meta['source_group']
            if not isinstance(group,str) or not group.strip(): raise ValueError('Missing source group')
            boxes = read_boxes(label,['cutting_board'])
            original = meta['boxes_normalized']
            if len(boxes) != len(original): raise ValueError('Label count differs from reviewed boxes')
            for (_,cx,cy,w,h), box in zip(boxes,original):
                if len(box)!=4 or not all(abs(a-b)<1e-6 for a,b in zip([cx-w/2,cy-h/2,cx+w/2,cy+h/2],box)):
                    raise ValueError('Labels differ from reviewed boxes')
            decision=meta['review_decision']
            if (decision=='all_boards_boxed' and not boxes) or (decision=='confirmed_no_board' and boxes) or decision not in ('all_boards_boxed','confirmed_no_board'):
                raise ValueError('Whole-image review decision contradicts labels')
            with Image.open(image) as im:
                rgb=im.convert('RGB')
                if list(im.size)!=meta['size']: raise ValueError('Image size differs from metadata')
                pixel=hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode()+rgb.tobytes()).hexdigest()
            records.append({'id':folder.name,'folder':str(folder),'image':str(image),'label':str(label),
                            'source_group':group,'boxes':len(boxes),'image_sha256':digest,'label_sha256':sha(label),
                            'metadata_sha256':sha(folder/'metadata.json'),'pixels_sha256':pixel,'dhash':dhash(image)})
        except Exception as exc: errors.append({'folder':str(folder),'error':str(exc)})
    duplicates,near=[],[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if a['pixels_sha256']==b['pixels_sha256']:
                duplicates.append({'a':a['id'],'b':b['id'],'same_group':a['source_group']==b['source_group']})
            elif (a['dhash']^b['dhash']).bit_count()<=4 and a['source_group']!=b['source_group']:
                near.append({'a':a['id'],'b':b['id'],'distance':(a['dhash']^b['dhash']).bit_count()})
    fingerprint=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    counts=Counter(r['source_group'] for r in records)
    return {'pool':str(pool),'records':records,'errors':errors,'incomplete_ignored':incomplete,
            'groups':dict(counts),'exact_duplicates':duplicates,'cross_group_near_candidates':near,
            'fingerprint':fingerprint,'note':'No claim of freshness against historical/pretraining data; manual labels still need quality review.'}


def build_dataset(pool, assignment_file, destination):
    audit=pool_audit(pool)
    assignment=json.loads(Path(assignment_file).read_text(encoding='utf-8-sig'))
    if audit['errors']: raise ValueError('Pool has invalid records; inspect audit first')
    if assignment.get('pool_fingerprint')!=audit['fingerprint']: raise ValueError('Pool changed since assignment template; audit again')
    choices=assignment.get('groups',{})
    if set(choices)!=set(audit['groups']): raise ValueError('Assignment must cover exactly all source groups')
    if any(s not in ('train','val','test','exclude') for s in choices.values()): raise ValueError('Assign each group train/val/test/exclude explicitly')
    if not assignment.get('reviewer','').strip() or not assignment.get('review_notes','').strip():
        raise ValueError('Record reviewer and review notes before building')
    records=[r for r in audit['records'] if choices[r['source_group']]!='exclude']
    selected={r['id']:r for r in records}
    for pair in audit['exact_duplicates']:
        if pair['a'] in selected and pair['b'] in selected: raise ValueError('Duplicate selected images: exclude duplicate group or correct pool')
    for pair in audit['cross_group_near_candidates']:
        if pair['a'] in selected and pair['b'] in selected and choices[selected[pair['a']]['source_group']]!=choices[selected[pair['b']]['source_group']]:
            raise ValueError('Near-duplicate candidates cross splits; review source grouping first')
    for split in ('train','val','test'):
        subset=[r for r in records if choices[r['source_group']]==split]
        if not subset or not any(r['boxes'] for r in subset): raise ValueError(f'{split} requires images and positive board labels')
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    groups={}
    for r in records:
        # Recheck the source hashes immediately before copying; no symlinks/hardlinks to mutable labels.
        if sha(r['image'])!=r['image_sha256'] or sha(r['label'])!=r['label_sha256'] or sha(Path(r['folder'])/'metadata.json')!=r['metadata_sha256']:
            raise ValueError('Pool changed during build')
        split=choices[r['source_group']]
        image=destination/'images'/split/(r['id']+'.png')
        label=destination/'labels'/split/(r['id']+'.txt')
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(r['image'],image);shutil.copyfile(r['label'],label)
        groups[str(image)]=r['source_group']
    data={'path':str(destination),'train':'images/train','val':'images/val','test':'images/test','names':['cutting_board']}
    (destination/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    dump(destination/'scene_groups.json',groups)
    dump(destination/'source_manifest.json',{'assignment':assignment,'pool_audit':audit})
    final=audit_dataset(destination/'data.yaml',groups_file=destination/'scene_groups.json')
    dump(destination/'data_audit.json',final)
    if not final['passed']: raise ValueError('Dataset audit failed; no READY marker or training plan created')
    dump(destination/'experiment_draft.json',{
        'experiment_name':'manual_boards_'+destination.name,'data':str(destination/'data.yaml'),
        'initial_weights':str(BASE/'training_runs/board_v2_real/weights/best.pt'),
        'groups_file':str(destination/'scene_groups.json'),'data_status':'manual_development_not_independent',
        'allow_training':False,'training':{'epochs':40,'imgsz':640,'batch':8,'device':0,'workers':0,'amp':False,'seed':42},
        'evaluation':{'confidence':0.5,'match_iou':0.5},
        'description':'Draft only. Set acceptance rules before training; assess source quality and history overlap. Do not call held-out groups independent site performance without evidence.'})
    (destination/'READY').write_text('Dataset integrity checks passed. Training remains disabled in draft.\n',encoding='utf-8')
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',default=str(BASE/'annotation_pool'))
    parser.add_argument('--output',default=None)
    parser.add_argument('--assignment',help='Explicit group assignments; omit to audit only')
    args=parser.parse_args()
    destination=Path(args.output) if args.output else BASE/'annotation_audits'/uuid.uuid4().hex
    if args.assignment:
        result=build_dataset(args.pool,args.assignment,destination)
        print(f'Dataset built; {sum(c["images"] for c in result["counts"].values())} images. Training not started. {destination}')
    else:
        report=pool_audit(args.pool)
        destination.mkdir(parents=True,exist_ok=False)
        dump(destination/'pool_audit.json',report)
        dump(destination/'group_assignment.template.json',{'pool_fingerprint':report['fingerprint'],
            'groups':{g:None for g in report['groups']},'reviewer':'','review_notes':''})
        (destination/'说明.txt').write_text(f'有效样本 {len(report["records"])}，来源组 {len(report["groups"])}，错误 {len(report["errors"])}。\n先审核来源和标注，再在模板中按组填写 train/val/test/exclude。没有自动拆分或训练。\n',encoding='utf-8')
        print(f'Audit: {len(report["records"])} valid records; {len(report["errors"])} errors. {destination}')

if __name__=='__main__': main()
