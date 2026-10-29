"""Frozen-prompt diagnostic of every YOLO proposal and geometry-excluded food.

Standalone opinions only: never relabel detections or assert physical contact.
"""
import argparse, hashlib, json, math
from pathlib import Path
from PIL import Image, ImageDraw
from food_category_review import FOOD_CATEGORY_PROMPT, CATEGORIES

B = Path(__file__).resolve().parent
SOURCE = B / 'food_batch36_v2'
OUT = B / 'food_candidate_review_v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def prepare():
    OUT.mkdir(exist_ok=False)
    cases = json.loads((SOURCE/'locked_plan.json').read_text(encoding='utf-8'))['cases']
    yolo = json.loads((SOURCE/'yolo_food.json').read_text(encoding='utf-8'))['rows']
    vlm = json.loads((SOURCE/'results.json').read_text(encoding='utf-8'))['rows']
    entries = []
    for i, case in enumerate(cases):
        assert sha(case['image']) == case['sha256']
        proposals = [('yolo', j, r) for j, r in enumerate(yolo[i]['predictions'])]
        proposals += [('geometry_excluded', j, r) for j, r in enumerate(vlm[i].get('regions', [])) if r['location'] != 'board_candidate']
        with Image.open(case['image']) as im:
            for origin, j, region in proposals:
                x1,y1,x2,y2 = region['box']; w=x2-x1;h=y2-y1
                crop = [max(0, math.floor(x1-.1*w)), max(0, math.floor(y1-.1*h)), min(im.width, math.ceil(x2+.1*w)), min(im.height, math.ceil(y2+.1*h))]
                key=f'{i:02d}_{origin}_{j:02d}'
                target=OUT/f'{key}.jpg'; im.crop(crop).convert('RGB').save(target, quality=95)
                entries.append(dict(id=key, frame_index=i, source=case['source'], seconds=case['seconds'], image=case['image'], image_sha256=case['sha256'], origin=origin, original_region=region, crop_xyxy=crop, crop=str(target), crop_sha256=sha(target), training_eligible=False, physical_contact_verified=False))
    save(OUT/'locked_plan.json', dict(entries=entries, prompt=FOOD_CATEGORY_PROMPT, max_new_tokens=12, margin=.1, model_frozen=True, blind_test=False, selection='All YOLO proposals plus all geometry-excluded food from the fixed 36-frame batch; no selection by new classifier response.'))
    for page in range((len(entries)+11)//12):
        sheet=Image.new('RGB',(1200,1040),'white');draw=ImageDraw.Draw(sheet)
        for k,e in enumerate(entries[page*12:(page+1)*12]):
            im=Image.open(e['crop']);im.thumbnail((290,215));x=(k%4)*300;y=(k//4)*345
            sheet.paste(im,(x,y));draw.text((x+3,y+220),e['id'],fill='black');draw.text((x+3,y+240),e['source'],fill='black')
            draw.text((x+3,y+260),e['original_region'].get('category',e['original_region'].get('location','')),fill='black')
        sheet.save(OUT/f'candidates_{page}.jpg')
    print('Prepared',len(entries),'candidates',flush=True)

def run():
    from region_vlm_runtime import LocalRegionModel
    plan=json.loads((OUT/'locked_plan.json').read_text(encoding='utf-8'))
    if (OUT/'results.json').exists() or (OUT/'partial_results.json').exists():
        raise FileExistsError('Preserve previous inference; choose a new experiment directory.')
    model=LocalRegionModel(); rows=[]
    for e in plan['entries']:
        assert sha(e['image'])==e['image_sha256'] and sha(e['crop'])==e['crop_sha256']
        r=dict(e, review_status='needs_review', opinion='unknown', detection_removed=False)
        try:
            # Same original-pixel crop as the existing category function; JPEG is display only.
            im=Image.open(e['image']).convert('RGB').crop(e['crop_xyxy'])
            raw=model.predict(im,prompt=plan['prompt'],max_new_tokens=12)
            r['raw_response']=raw
            r['opinion']=raw.strip().lower() if raw.strip().lower() in CATEGORIES else 'unknown'
            r['response_valid']=raw.strip().lower() in CATEGORIES
        except Exception as exc:
            r['error']=repr(exc);r['response_valid']=False
        original=e['original_region'].get('category')
        r['comparison']='not_comparable' if not original or r['opinion'] in ('unknown','other') else ('same_label' if original==r['opinion'] else 'different_label')
        rows.append(r);save(OUT/'partial_results.json',dict(rows=rows,model_revision=model.revision))
        print(e['id'],r['opinion'],flush=True)
    save(OUT/'results.json',dict(rows=rows,model_revision=model.revision,training_performed=False,plan_sha256=sha(OUT/'locked_plan.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.action=='prepare' else run()
