import json,zipfile,os,shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import yaml
from prepare_lvis_boards import fetch,prior_photo_ids,flickr_id
from experiment_data import audit_dataset,sha256,dhash
BASE=Path(__file__).resolve().parent;RAW=BASE.parent/'work/datasets/lvis';OUT=BASE/'board_lvis_expanded_train'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    with zipfile.ZipFile(RAW/'lvis_v1_train.json.zip') as z:d=json.loads(z.read(next(n for n in z.namelist() if n.endswith('.json'))))
    ann=defaultdict(list)
    for a in d['annotations']:
        if a['category_id']==253:ann[a['image_id']].append(a)
    used=prior_photo_ids();candidates=[];excluded=[]
    for im in d['images']:
        if im['license']!=5 or not ann[im['id']] or 253 in im.get('not_exhaustive_category_ids',[]):continue
        if flickr_id(im['flickr_url']) in used:excluded.append({'id':im['id'],'reason':'prior Flickr ID'});continue
        candidates.append({'image':im,'annotations':ann[im['id']],'flickr_photo_id':flickr_id(im['flickr_url'])})
    del d,ann
    print('eligible',len(candidates),flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool: downloaded=list(pool.map(fetch,candidates))
    index=json.loads((RAW/'prior_image_index.json').read_text())
    current=json.loads((BASE/'experiments/board_capacity_s_validation/data_audit.json').read_text())
    index += [{'file':r['image'],'sha256':r['image_sha256'],'dhash':r['dhash']} for r in current['records']]
    keep=[]
    for record in downloaded:
        if record.get('error'):excluded.append({'id':record['image']['id'],'reason':record['error']});continue
        matches=[r for r in index if record['sha256']==r['sha256'] or (record['dhash']^r['dhash']).bit_count()<=4]
        if matches:excluded.append({'id':record['image']['id'],'reason':'prior byte/near image','matches':[r['file'] for r in matches[:3]]});continue
        keep.append(record);index.append({'file':record['source_file'],'sha256':record['sha256'],'dhash':record['dhash']})
    if not keep:raise RuntimeError('No new eligible images')
    OUT.mkdir();old=BASE/'board_lvis_ccby'
    for split in ['train','val','test']:
        paths=[]
        for line in (old/f'{split}.txt').read_text(encoding='utf-8-sig').splitlines():
            src=Path(line);dst=OUT/'images'/split/src.name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
            label=OUT/'labels'/split/src.with_suffix('.txt').name;label.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(old/'labels'/split/label.name,label);paths.append(str(dst))
        if split=='train':
            for record in keep:
                im=record['image'];name=f'sa_{im["id"]:012d}'
                dst=OUT/'images/train'/f'{name}.jpg';os.link(record['source_file'],dst);paths.append(str(dst))
                lines=[]
                for a in record['annotations']:
                    x,y,w,h=a['bbox'];iw,ih=im['width'],im['height'];x1,y1=max(0,x),max(0,y);x2,y2=min(iw,x+w),min(ih,y+h)
                    if x2<=x1 or y2<=y1:raise ValueError('Invalid annotation box')
                    lines.append(f'0 {(x1+x2)/2/iw:.8f} {(y1+y2)/2/ih:.8f} {(x2-x1)/iw:.8f} {(y2-y1)/ih:.8f}')
                (OUT/'labels/train'/f'{name}.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        (OUT/f'{split}.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
    (OUT/'data.yaml').write_text(yaml.safe_dump({'path':str(OUT),'train':'train.txt','val':'val.txt','test':'test.txt','names':['cutting_board']},sort_keys=False),encoding='utf-8')
    manifest={'new_train_records':keep,'excluded':excluded,'added_images':len(keep),'added_boxes':sum(len(r['annotations']) for r in keep),
              'license_url':'https://creativecommons.org/licenses/by-sa/2.0/','license_id':5,'original_split':'LVIS train',
              'scope':'local development only; no image/model distribution in this experiment',
              'limitations':['Source metadata supplies license and Flickr URL, not a verified author identity.','Hashes/dHash cannot establish independent scenes or pretraining novelty.','Existing validation and test remain held out; validation already reused for development.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    audit=audit_dataset(OUT/'data.yaml');(OUT/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if not audit['passed']:raise RuntimeError('Audit failed; do not train')
    print(json.dumps({'added_images':len(keep),'added_boxes':manifest['added_boxes'],'counts':audit['counts'],'audit_passed':audit['passed']}),flush=True)

if __name__=='__main__':main()
