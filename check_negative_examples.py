"""Small visually screened diagnostic set; not an independent benchmark."""
import csv,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work'/'yolo-config')

def main():
    from ultralytics import YOLO
    candidates=json.loads((BASE/'negative_candidates.json').read_text())
    selected=[Path(candidates[i]) for i in [10,12,13,16,22]]
    reference=json.loads((BASE/'vegetable_v2_reference.json').read_text())['weights']
    comparison=json.loads((BASE/'vegetable_refinement_metrics.json').read_text())
    records={}
    for name,weights in [('reference',reference),('candidate',comparison['candidate_weights'])]:
        model=YOLO(weights);results=[]
        for file in selected:
            pred=model.predict(str(file),imgsz=768,conf=.5,device=0,verbose=False)[0]
            results.append({'image_id':file.stem,'detections':len(pred.boxes),'confidences':pred.boxes.conf.cpu().tolist(),
                'boxes':pred.boxes.xyxy.cpu().tolist()})
        records[name]={'images_with_any_detection':sum(r['detections']>0 for r in results),'images':len(results),'results':results}
    ids={p.stem for p in selected}
    attribution=[r for r in csv.DictReader((BASE/'board_dataset'/'attribution.csv').open(encoding='utf-8-sig')) if r['ImageID'] in ids]
    report={'definition':'No visible culinary vegetables, screened from contact sheet: bread, cheese, knives, empty board.',
        'selection':'Five board-training images absent from the vegetable dataset. Visual screening by assistant; not independent expert labeling.',
        'confidence':.5,'imgsz':768,'results':records,'attribution':attribution,
        'limitation':'Five curated images are a qualitative diagnostic, not a kitchen false-alarm-rate estimate.'}
    (BASE/'negative_example_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({name:{k:v for k,v in r.items() if k!='results'} for name,r in records.items()},indent=2))

if __name__=='__main__':main()
