"""Replay saved combined geometry only; no inference or historical result overwrite."""
import json, hashlib
from pathlib import Path
from PIL import Image
from review_combined import position_evidence
from food_region_gate import locate
BASE=Path(__file__).resolve().parent
out=BASE/'experiments/fusion_geometry_reaudit_v1'
out.mkdir(exist_ok=False)
rows=[]
for path in (BASE/'experiments').rglob('result.json'):
    data=json.loads(path.read_text(encoding='utf-8'))
    if not all(k in data for k in ['board_source','board_presence','board_boxes','food_from_detector','image']):
        continue
    image=Path(data['image'])
    if not image.is_file():
        continue
    if hashlib.sha256(image.read_bytes()).hexdigest()!=data.get('sha256'):
        raise ValueError(f'Image hash mismatch: {path}')
    with Image.open(image) as im: w,h=im.size
    quality,boxes,usable=position_evidence(data['board_boxes'],w,h,data['board_source'],data['board_presence'])
    regions=[locate(f['box'],boxes or data['board_boxes'],[],usable) for f in data['food_from_detector']]
    rows.append({'result':str(path),'sha256':data['sha256'],'before':data['position_usable'],'after':usable,
                 'reason':quality['reason'],'old_locations':[f['location'] for f in data['food_regions']],
                 'new_locations':[f['location'] for f in regions]})
summary={'saved_results':len(rows),'position_changed':sum(r['before']!=r['after'] for r in rows),
         'food_location_changed':sum(r['old_locations']!=r['new_locations'] for r in rows),
         'limitation':'Saved individual results only, not full five-batch replay; no accuracy claim.'}
(out/'results.json').write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
