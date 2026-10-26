"""New-source positive-board probe from official VISOR val; no detector used for sampling."""
import io,json,re,zipfile,requests,hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw
from experiment_data import sha256,dhash
BASE=Path(__file__).resolve().parent;RAW=BASE.parent/'work/datasets/visor';OUT=BASE/'visor_board_probe'
URL='https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/'

class RangeFile(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0
        r=requests.get(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'},timeout=40);r.raise_for_status()
        if r.status_code!=206:raise ValueError('Server does not support ranges')
        self.size=int(r.headers['Content-Range'].split('/')[-1]);self.etag=r.headers.get('ETag')
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        if n>8*1024*1024:raise ValueError('Range read too large')
        headers={'Range':f'bytes={self.pos}-{self.pos+n-1}','Accept-Encoding':'identity'}
        if self.etag:headers['If-Range']=self.etag
        r=requests.get(self.url,headers=headers,timeout=45);r.raise_for_status()
        if r.status_code!=206 or len(r.content)!=n:raise ValueError('Invalid ranged response')
        if self.etag and r.headers.get('ETag')!=self.etag:raise ValueError('Archive changed')
        self.pos+=n;return r.content

def participant_pick(item):
    participant,videos=item;notes=[]
    for video in sorted(videos)[:3]:
        path=RAW/(video+'.json')
        try:
            if not path.exists():
                r=requests.get(URL+f'GroundTruth-SparseAnnotations/annotations/val/{video}.json',timeout=45);r.raise_for_status()
                if len(r.content)>50*1024*1024:raise ValueError('Annotation too large')
                path.write_bytes(r.content)
            data=json.loads(path.read_text(encoding='utf-8'))
            for row in sorted(data['video_annotations'],key=lambda r:r['image']['name']):
                boards=[a for a in row['annotations'] if a['class_id']==18]
                if not boards or any(a.get('exhaustive')!='y' or not a.get('segments') for a in boards):continue
                valid=[]
                for a in boards:
                    pts=[point for segment in a['segments'] for point in segment]
                    if not pts:break
                    x1=max(0,min(p[0] for p in pts));x2=min(1920,max(p[0] for p in pts));y1=max(0,min(p[1] for p in pts));y2=min(1080,max(p[1] for p in pts))
                    if x2-x1<10 or y2-y1<10:break
                    valid.append([x1/1920,y1/1080,x2/1920,y2/1080])
                if len(valid)==len(boards):return {'participant':participant,'video':video,'frame':row['image'],'annotations':boards,'boxes_normalized':valid,'annotation_file':str(path),'annotation_sha256':sha256(path)}
            notes.append({'video':video,'reason':'No eligible exhaustive board frame'})
        except Exception as exc:notes.append({'video':video,'error':str(exc)})
    return {'participant':participant,'excluded':notes}

def download_frame(record):
    if 'excluded' in record:return record
    try:
        video=record['video'];participant=record['participant'];name=record['frame']['name']
        url=URL+f'GroundTruth-SparseAnnotations/rgb_frames/val/{participant}/{video}.zip'
        with zipfile.ZipFile(RangeFile(url)) as z:
            matches=[n for n in z.namelist() if Path(n).name==name]
            if len(matches)!=1:raise ValueError('Frame not unique in ZIP')
            info=z.getinfo(matches[0])
            if info.file_size>10*1024*1024:raise ValueError('Frame too large')
            data=z.read(matches[0])
        path=RAW/'frames'/name;path.parent.mkdir(exist_ok=True);path.write_bytes(data)
        with Image.open(path) as im:im.verify()
        return {**record,'source_image':str(path),'download_url':url,'zip_member':matches[0],'image_sha256':sha256(path),'dhash':dhash(path)}
    except Exception as exc:return {**record,'download_error':str(exc)}

def main():
    OUT.mkdir(exist_ok=False);RAW.mkdir(exist_ok=True)
    plan={'source':URL,'scope':'New-source positive-only board localization probe, not kitchen misuse/negative false-alarm evaluation',
          'selection':'Official val, up to first3 lexical videos per participant; first exhaustive-y class18 frame with all visible boxes >=10px; at most one frame per participant. No model-based selection.',
          'coordinate_basis':'Sparse polygons use1920x1080 per official VISOR-VIS vis.py; normalize then scale to actual image dimensions.',
          'license':'CC BY-NC4.0; local noncommercial research probe only; no training or publication of images/weights.',
          'max_participants':21,'confidence':.5,'match_iou':.5,'imgsz':640,'no_training':True,'no_threshold_sweep':True,
          'models':{'default_board':str(BASE/'training_runs/board_v2_real/weights/best.pt'),'previous_s':str(BASE/'experiments/board_capacity_s_validation/training/weights/best.pt'),'expanded_s':str(BASE/'experiments/board_expanded_s_reviewed/training/weights/best.pt')}}
    plan['model_sha256']={k:sha256(v) for k,v in plan['models'].items()}
    (OUT/'locked_sampling_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    listing=requests.get(URL+'GroundTruth-SparseAnnotations/annotations/val/',timeout=40);listing.raise_for_status()
    names=re.findall(r'href="(P\d+_\d+\.json)"',listing.text);groups=defaultdict(list)
    for name in names:groups[name.split('_')[0]].append(name[:-5])
    chosen=sorted(groups.items())[:21]
    with ThreadPoolExecutor(max_workers=5) as pool:records=list(pool.map(participant_pick,chosen))
    (OUT/'annotation_selection.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('annotation_selected',sum('video' in r for r in records),'of',len(chosen),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(download_frame,records))
    old=json.loads((BASE.parent/'work/datasets/lvis/prior_image_index.json').read_text())
    for audit_path in [BASE/'experiments/board_expanded_s_reviewed/data_audit.json']:
        audit=json.loads(audit_path.read_text());old += [{'file':r['image'],'sha256':r['image_sha256'],'dhash':r['dhash']} for r in audit['records']]
    kept=[];excluded=[]
    for r in records:
        if 'excluded' in r or 'download_error' in r:excluded.append(r);continue
        matches=[a for a in old if a['sha256']==r['image_sha256'] or (a['dhash']^r['dhash']).bit_count()<=4]
        if matches:excluded.append({**r,'overlap_matches':[a['file'] for a in matches]});continue
        image=OUT/'images'/r['frame']['name'];image.parent.mkdir(exist_ok=True);image.write_bytes(Path(r['source_image']).read_bytes())
        label=OUT/'labels'/image.with_suffix('.txt').name;label.parent.mkdir(exist_ok=True)
        lines=[f'0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}' for x1,y1,x2,y2 in r['boxes_normalized']]
        label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        kept.append({**r,'image':str(image),'label':str(label),'label_sha256':sha256(label)})
    manifest={'records':kept,'excluded':excluded,'images':len(kept),'boards':sum(len(r['boxes_normalized']) for r in kept),'participant_groups':len({r['participant'] for r in kept}),
              'limitations':['Positive-only selection; no negative-scene false-alarm rate.','One frame per participant limits temporal duplicates but does not prove all households independent.','Visible-mask boxes differ from full-object boxes when occluded.','New to this project, but generic pretraining novelty not proved.','Metadata-first selection is not random sampling or surveillance-domain representativeness.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(1280,max(1,(len(kept)+3)//4)*220),'white')
    for i,r in enumerate(kept):
        im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
        for x1,y1,x2,y2 in r['boxes_normalized']:draw.rectangle((x1*im.width,y1*im.height,x2*im.width,y2*im.height),outline='orange',width=4)
        im.thumbnail((315,190));x=i%4*320;y=i//4*220;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+2,y+195),r['video'],fill='black')
    sheet.save(OUT/'labels_preview.jpg')
    print(json.dumps({k:manifest[k] for k in ['images','boards','participant_groups']}),flush=True)

if __name__=='__main__':main()
