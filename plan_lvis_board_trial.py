"""Write the trial criteria before predictions on the fresh LVIS holdout."""
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent

def main():
    rule=BASE/'lvis_board_decision_rule.json'
    plan=BASE/'lvis_board_plan.json'
    if rule.exists() or plan.exists():raise RuntimeError('Preserve the existing experiment plan')
    data=BASE/'board_lvis_ccby'
    registry=json.loads((BASE/'active_models.json').read_text(encoding='utf-8-sig'))
    definition={'created_before_model_evaluation':True,'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'question':'Does new, separately annotated LVIS board data improve the current board detector on fresh source images?',
        'reference_weights':registry['board']['weights'],'fresh_data':str(data/'data.yaml'),
        'confidence':.5,'match_iou':.5,
        'all_required':{'fresh_min_precision':.70,'fresh_min_recall':.50,'fresh_min_map50':.60,
            'fresh_min_f1_improvement':.05,'fresh_max_precision_drop':.03,'fresh_max_recall_drop':.03,
            'fresh_max_extra_false_boxes_per_negative_image':.05,
            'historical_max_map50_drop':.03,'historical_max_fixed_recall_drop':.05},
        'if_fails':'Keep the currently registered board detector; retain the candidate and data for research.',
        'scope':'Public household/food photos. Not a new canteen/surveillance benchmark; small fresh test set.',
        'provenance':'LVIS v1 chopping_board 253, image license ID4 only; no not-exhaustive labels; prior Flickr IDs/content filtered before training.'}
    rule.write_text(json.dumps(definition,ensure_ascii=False,indent=2),encoding='utf-8')
    payload={'experiment_name':'lvis_board_ccby_trial','data':str(data/'data.yaml'),
        'initial_weights':registry['board']['weights'],'data_status':'fresh_holdout_declared','allow_training':True,
        'near_duplicate_hamming':4,
        'training':{'epochs':50,'imgsz':640,'batch':8,'device':0,'workers':0,'amp':False,'seed':42,
            'optimizer':'AdamW','lr0':.0005,'lrf':.1,'freeze':0,'mosaic':.3,'scale':.2,'close_mosaic':10},
        'evaluation':{'confidence':.5,'match_iou':.5},
        'acceptance':{'min_map50':.60,'min_map50_95':.30,'min_precision':.70,'min_recall':.50},
        'external_relative_decision_rule':str(rule),'external_relative_decision_rule_sha256':hashlib.sha256(rule.read_bytes()).hexdigest(),
        'description':'New LVIS train/val photos after old-photo/content exclusion. No owner/scene grouping is available; source-scene independence is not proven.'}
    plan.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    # Preserve image license/source metadata next to the derived data, including historical rows.
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    rows=[];historical={Path(r['source_file']).stem for r in manifest['records'] if r['source']=='historical_Open_Images_training_only'}
    known={}
    for file in BASE.rglob('attribution.csv'):
        if data in file.parents:continue
        for r in csv.DictReader(file.open(encoding='utf-8-sig')):
            if r.get('ImageID') in historical:known[r['ImageID']]=r
    for r in manifest['records']:
        if r['source']=='LVIS_v1':
            rows.append({'image_id':str(r['image']['id']),'source':'LVIS_v1/COCO','split':r['split'],
                'file':r['image_file'],'flickr_url':r['image']['flickr_url'],'photo_page':r['original_photo_page'],
                'license_url':r['license_url'],'author':'not supplied in LVIS metadata',
                'annotation_license':'https://creativecommons.org/licenses/by/4.0/','attribution_note':r['attribution_note']})
        else:
            old=known.get(Path(r['source_file']).stem,{})
            rows.append({'image_id':Path(r['source_file']).stem,'source':'Open Images historical train','split':'train',
                'file':r['image'],'flickr_url':old.get('OriginalURL',''),'photo_page':old.get('OriginalLandingURL',''),
                'license_url':old.get('License',''),'author':old.get('Author','not supplied'),
                'annotation_license':'https://creativecommons.org/licenses/by/4.0/','attribution_note':'Retained historical training image; not a fresh holdout.'})
    with (data/'attribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(plan,flush=True)

if __name__=='__main__':main()
