"""Counterfactual policy illustration on public images; not a misuse benchmark."""
from pathlib import Path
import json
import os
import csv
from check_board_misuse import inspect_image, selected_weights, food_image_size
from misuse_rules import BoardRule,Detection,evaluate

BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work'/'yolo-config')

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',default='',help='Prefix for new output folders; preserves prior demo when set')
    args=parser.parse_args()
    from ultralytics import YOLO
    board_model=YOLO(str(selected_weights('baseline_metrics.json','training_runs/board_baseline/weights/best.pt')))
    vegetable_model=YOLO(str(selected_weights('vegetable_metrics.json','training_runs/vegetable_baseline/weights/best.pt')))
    files=sorted((BASE/'board_dataset'/'images'/'test').glob('*.jpg'))
    chosen=None
    for file in files:
        boards=board_model.predict(str(file),conf=.5,device=0,verbose=False)[0]
        foods=vegetable_model.predict(str(file),conf=.5,imgsz=food_image_size(),device=0,verbose=False)[0]
        rules=[BoardRule(f'candidate_{i}',tuple(box),frozenset({'meat'})) for i,box in enumerate(boards.boxes.xyxy.cpu().tolist(),1)]
        detections=[Detection('vegetable',float(c),tuple(b)) for b,c in zip(foods.boxes.xyxy.cpu().tolist(),foods.boxes.conf.cpu().tolist())]
        if any(r['status']=='suspected_misuse' for r in evaluate(rules,detections)):
            chosen=file;break
    if chosen is None:
        chosen=files[0]
        print('No automatic conflict illustration found at fixed thresholds; saving an inconclusive example.',flush=True)
    for name,allowed in [('demo_meat_only',['meat']),('demo_vegetable_only',['vegetable'])]:
        dest=BASE/'misuse_results'/(args.prefix+name)
        config={'demo_not_site_policy':True,
            'description':'Public image with an invented policy for software illustration; not a real observed violation or labeled misuse test.',
            'all_detected_boards_allowed_foods':allowed,'board_confidence':.5,'food_confidence':.5,'min_food_overlap':.8}
        report=inspect_image(chosen,config,dest)
        (dest/'rules.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        with (BASE/'board_dataset'/'attribution.csv').open(encoding='utf-8-sig') as f:
            attribution=next(r for r in csv.DictReader(f) if r['ImageID']==chosen.stem)
        attribution['Derivative']='evidence.jpg: original image plus model predictions'
        (dest/'attribution.json').write_text(json.dumps(attribution,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,chosen.name,report['overall_status'],flush=True)

if __name__=='__main__':main()
