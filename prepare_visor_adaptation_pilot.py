"""Participant-disjoint VISOR adaptation data; no model-assisted labels or selection."""
import json,re,os,zipfile,requests,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import yaml
from PIL import Image,ImageDraw
from prepare_visor_probe import RangeFile,URL
from experiment_data import audit_dataset,sha256,dhash
BASE=Path(__file__).resolve().parent;RAW=BASE.parent/'work/datasets/visor';OUT=BASE/'visor_adaptation_pilot'

def metadata_pick(item):
    participant,videos=item;records=[];notes=[]
    for video in sorted(videos)[:8]:
        path=RAW/('train_'+video+'.json')
        try:
            if not path.exists():
                response=requests.get(URL+f'GroundTruth-SparseAnnotations/annotations/train/{video}.json',timeout=45);response.raise_for_status();path.write_bytes(response.content)
            data=json.loads(path.read_text(encoding='utf-8'));used=set();last=-1000000;video_records=[]
            for row in sorted(data['video_annotations'],key=lambda row:row['image']['name']):
                boards=[a for a in row['annotations'] if a['class_id']==18]
                if not boards or any(a.get('exhaustive')!='y' for a in boards):continue
                frame=int(re.search(r'frame_(\d+)',row['image']['name']).group(1));sequence=row['image']['subsequence']
                if frame-last<300:continue
                boxes=[]
                for a in boards:
                    points=[p for seg in a['segments'] for p in seg]
                    if not points:break
                    x1=max(0,min(p[0] for p in points));y1=max(0,min(p[1] for p in points));x2=min(1920,max(p[0] for p in points));y2=min(1080,max(p[1] for p in points))
                    if x2-x1<20 or y2-y1<20:break
                    boxes.append([x1/1920,y1/1080,x2/1920,y2/1080])
                if len(boxes)!=len(boards):continue
                video_records.append({'participant':participant,'video':video,'frame':row['image'],'boxes_normalized':boxes,'annotation_file':str(path),'annotation_sha256':sha256(path)})
                used.add(sequence);last=frame
                if len(video_records)>=30:break
            records+=video_records
            notes.append({'video':video,'eligible_selected':len(video_records)})
        except Exception as exc:notes.append({'video':video,'error':str(exc)})
    return {'participant':participant,'records':records[:40],'notes':notes}

def fetch_video(item):
    video,records=item;out=[]
    participant=records[0]['participant'];url=URL+f'GroundTruth-SparseAnnotations/rgb_frames/train/{participant}/{video}.zip'
    try:
        with zipfile.ZipFile(RangeFile(url)) as z:
            lookup={Path(n).name:n for n in z.namelist()}
            for record in records:
                try:
                    name=record['frame']['name'];member=lookup[name];path=RAW/'adaptation_frames'/name;path.parent.mkdir(exist_ok=True)
                    if not path.exists():path.write_bytes(z.read(member))
                    with Image.open(path) as image:image.verify()
                    out.append({**record,'source_image':str(path),'image_sha256':sha256(path),'dhash':dhash(path),'download_url':url,'zip_member':member})
                except Exception as exc:out.append({**record,'error':str(exc)})
    except Exception as exc:out=[{**r,'error':str(exc)} for r in records]
    return out

def main():
    OUT.mkdir(exist_ok=False)
    listing=json.loads((BASE/'visor_train_source_listing.json').read_text(encoding='utf-8'))
    assign={'P10':'train','P11':'train','P15':'val','P35':'test'}
    plan={'participants':assign,'selection':'Official train; first8 lexical videos; exhaustive-y board class18; bbox sides>=20 pixels at1920x1080; frame-number gap>=300, max30/video and40/participant. Deterministic annotation-only selection.',
          'split_revision_reason':'Initial metadata-only assignment had empty validation. Revised before model training/output based solely on annotation availability; only4 eligible participants. 2train/1val/1test is a small pilot, not strong generalization evidence.','split_seed':20261026,'old_official_val_participants_excluded':True,'mixing':'Retain526 existing training images only; new val/test are VISOR participants, no old validation/test mixed into training.',
          'purpose':'Local noncommercial research, CC BY-NC4.0; no image or weight publication; positive-only new source; no claimed scene-negative rate.'}
    (OUT/'locked_sampling_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    with ThreadPoolExecutor(max_workers=5) as pool:selected=list(pool.map(metadata_pick,[(p,listing['groups'][p]) for p in sorted(assign)]))
    (OUT/'metadata_selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    videos=defaultdict(list)
    for group in selected:
        for record in group['records']:videos[record['video']].append(record)
    print('selected',sum(len(r) for r in videos.values()),'participant_counts',{g['participant']:len(g['records']) for g in selected},flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:downloaded=[r for group in pool.map(fetch_video,sorted(videos.items())) for r in group]
    prior=json.loads((BASE.parent/'work/datasets/lvis/prior_image_index.json').read_text())
    old=json.loads((BASE/'experiments/board_expanded_s_reviewed/data_audit.json').read_text())
    prior += [{'file':r['image'],'sha256':r['image_sha256'],'dhash':r['dhash']} for r in old['records']]
    for r in json.loads((BASE/'visor_board_probe/manifest.json').read_text())['records']:prior.append({'file':r['image'],'sha256':r['image_sha256'],'dhash':r['dhash']})
    records=[];excluded=[];paths={s:[] for s in ['train','val','test']};groups={}
    # Preserve previous training samples and their labels; no previous val/test copied.
    for row in old['records']:
        if row['split']!='train':continue
        src=Path(row['image']);dst=OUT/'images/train'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
        label=OUT/'labels/train'/dst.with_suffix('.txt').name;label.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['label'],label)
        paths['train'].append(str(dst));groups[str(dst)]='historical_train_unverified_scene_'+src.stem
    for record in downloaded:
        if record.get('error'):excluded.append(record);continue
        matches=[r for r in prior if r['sha256']==record['image_sha256'] or (r['dhash']^record['dhash']).bit_count()<=4]
        if matches:excluded.append({**record,'overlap':[r['file'] for r in matches[:3]]});continue
        split=assign[record['participant']];src=Path(record['source_image']);dst=OUT/'images'/split/src.name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
        label=OUT/'labels'/split/dst.with_suffix('.txt').name;label.parent.mkdir(parents=True,exist_ok=True)
        label.write_text('\n'.join(f'0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}' for x1,y1,x2,y2 in record['boxes_normalized'])+'\n',encoding='utf-8')
        paths[split].append(str(dst));groups[str(dst)]=record['participant'];records.append({**record,'split':split,'image':str(dst),'label':str(label)})
    for split,items in paths.items():
        if not items:raise RuntimeError('Empty split; do not reassign after seeing model outputs')
        (OUT/f'{split}.txt').write_text('\n'.join(items)+'\n',encoding='utf-8')
    (OUT/'scene_groups.json').write_text(json.dumps(groups,indent=2),encoding='utf-8')
    (OUT/'data.yaml').write_text(yaml.safe_dump({'path':str(OUT),'train':'train.txt','val':'val.txt','test':'test.txt','names':['cutting_board']},sort_keys=False),encoding='utf-8')
    manifest={'new_records':records,'excluded':excluded,'participant_assignment':assign,'new_counts':{s:sum(r['split']==s for r in records) for s in paths},'limitation':'Historical train group names are placeholders, not verified scene IDs; new VISOR participants genuinely stay in one split. No prior val/test included in train.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    audit=audit_dataset(OUT/'data.yaml',groups_file=OUT/'scene_groups.json');(OUT/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if not audit['passed']:raise RuntimeError('Dataset audit failed; inspect before training')
    # Only training thumbnails are viewed before model training. Test remains held out.
    selected=[r for r in records if r['split']=='train']
    sheet=Image.new('RGB',(1280,max(1,(len(selected)+3)//4)*200),'white')
    for i,r in enumerate(selected):
        im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
        for x1,y1,x2,y2 in r['boxes_normalized']:draw.rectangle((x1*im.width,y1*im.height,x2*im.width,y2*im.height),outline='orange',width=4)
        im.thumbnail((315,170));x=i%4*320;y=i//4*200;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+2,y+175),r['frame']['name'],fill='black')
    sheet.save(OUT/'new_train_preview.jpg')
    print(json.dumps({'new_counts':manifest['new_counts'],'audit_passed':audit['passed'],'counts':audit['counts']}),flush=True)

if __name__=='__main__':main()
