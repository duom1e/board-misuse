"""Propose CC BY LVIS object crops. Not whole-image negative labels."""
import json,hashlib,random,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;RAW=B.parent/'work/datasets/lvis';E=B/'board_table_crops_v1'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    E.mkdir(exist_ok=True);(E/'images').mkdir(exist_ok=True)
    assert not (E/'manifest.json').exists()
    previous=json.loads((B/'board_lvis_ccby/manifest.json').read_text(encoding='utf-8'))['records']
    boards=[r for r in previous if r.get('split')=='train' and r.get('annotations') and r.get('image',{}).get('license')==4]
    random.Random(20261027).shuffle(boards);print('eligible boards',len(boards),flush=True)
    d=json.loads((RAW/'lvis_v1_val.json').read_text(encoding='utf-8'));by={}
    for a in d['annotations']:
        if a['category_id'] in [285,367,610,1050]:by.setdefault(a['image_id'],[]).append(a)
    used={int(p.stem) for p in (RAW/'images').glob('*.jpg') if p.stem.isdigit()}
    tables=[i for i in d['images'] if i['id'] in by and i['license']==4 and i['id'] not in used]
    random.Random(20261027).shuffle(tables)
    specs=[]
    for r in boards[:16]:specs.append({'metadata':r['image'],'annotations':r['annotations'],'class':'board','existing_path':r.get('source_file') or r.get('image_file'),'prior_exposure':'historical board training source'})
    for i in tables[:24]:specs.append({'metadata':i,'annotations':by[i['id']],'class':'table','prior_exposure':'new project source photo selected from official LVIS val'})
    def prepare(s):
        try:
            i=s['metadata'];p=Path(s['existing_path']) if s.get('existing_path') else RAW/'images'/f"{i['id']:012}.jpg"
            if not p.exists():
                path=i['coco_url'].split('images.cocodataset.org/')[-1];r=requests.get('https://s3.amazonaws.com/images.cocodataset.org/'+path,timeout=30);r.raise_for_status();p.parent.mkdir(exist_ok=True);p.write_bytes(r.content)
            im=Image.open(p).convert('RGB');a=max(s['annotations'],key=lambda a:a['bbox'][2]*a['bbox'][3]);x,y,w,h=a['bbox'];box=[max(0,math.floor(x)),max(0,math.floor(y)),min(im.width,math.ceil(x+w)),min(im.height,math.ceil(y+h))]
            assert box[2]-box[0]>=32 and box[3]-box[1]>=32
            crop=E/'images'/f"{s['class']}_{i['id']:012}.jpg";im.crop(box).save(crop,quality=95)
            return {'image':str(crop),'source_image':str(p),'source_sha256':digest(p),'sha256':digest(crop),'source_photo_id':i['id'],'class':s['class'],'crop_xyxy':box,'lvis_annotation_id':a['id'],'category_id':a['category_id'],'flickr_url':i['flickr_url'],'license':'CC BY 2.0 per LVIS metadata','prior_exposure':s['prior_exposure'],'review_status':'pending_crop_review'}
        except Exception as exc:return {'class':s['class'],'error':repr(exc),'source_photo_id':s['metadata']['id']}
    with ThreadPoolExecutor(max_workers=6) as pool:done=list(pool.map(prepare,specs))
    rows=[r for r in done if 'error' not in r]
    (E/'manifest.json').write_text(json.dumps({'records':rows,'errors':[r for r in done if 'error' in r],'training_ready':False,'scope':'Largest annotated object crop per photo. Class is target object identity, not whole-image board absence. Prior-pretraining exposure unknown; historical board sources already consumed.'},indent=2),encoding='utf-8')
    for start in range(0,len(rows),20):
        sheet=Image.new('RGB',(1000,5*180),(25,25,25));dr=ImageDraw.Draw(sheet)
        for j,r in enumerate(rows[start:start+20]):
            im=Image.open(r['image']);im.thumbnail((245,150));x=j%4*250;y=j//4*180;sheet.paste(im,(x,y));dr.text((x+3,y+152),f"{start+j} {r['class']} {r['source_photo_id']}",fill='white')
        sheet.save(E/f'review_{start//20}.jpg')
    print('prepared',len(rows),'errors',len(done)-len(rows),flush=True)
if __name__=='__main__':main()
