import json,hashlib,cv2
from pathlib import Path
from run_canteen_food_detector import evaluate
from ultralytics import YOLO
B=Path(__file__).resolve().parent;E=B/'experiments/bilibili_food_frozen_probe'
if __name__=='__main__':
 E.mkdir(exist_ok=True);d=json.loads((B/'bilibili_prep_manual_v1/manifest.json').read_text(encoding='utf-8'))
 assert d['review_status']=='assistant_food_overlay_reviewed_not_expert'
 rows=[r for r in d['records'] if r['localization_uncertainty']=='normal']
 source=json.loads((B/'bilibili_prep_screening/manifest.json').read_text(encoding='utf-8'))['source'];cap=cv2.VideoCapture(source);fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert fps>0
 for r in rows:r['seconds']=r['frame']/fps
 candidate=json.loads((B/'experiments/canteen_food_mixed_operational/decision.json').read_text())['selected_for_development']['weights']
 weights={'default_foodseg':B/'training_runs/foodseg_2cls/weights/best.pt','previous_mixed_candidate':Path(candidate)}
 plan={'models':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in weights.items()},'images':[r['image'] for r in rows],'conf':.5,'match_iou':.5,'imgsz':640,'annotation_manifest_sha256':hashlib.sha256((B/'bilibili_prep_manual_v1/manifest.json').read_bytes()).hexdigest(),'exclusions':'5 high-occlusion localization-uncertain frames excluded before model predictions. No thresholds/models tuned.','scope':'New-source foreground ROI diagnostic, 18 vegetable bulk-food images +2 no-obvious-bulk-food; no meat samples, same-video correlated, prior default predictions used in source screening; NOT independent final benchmark.','board_ground_truth':False,'training_performed':False}
 locked=E/'locked_plan.json'
 if locked.exists():assert json.loads(locked.read_text())==plan
 else:locked.write_text(json.dumps(plan,indent=2),encoding='utf-8')
 assert not (E/'comparison.json').exists(),'Do not re-evaluate a consumed diagnostic without a new declared plan.'
 results={k:evaluate(YOLO(str(p)),rows) for k,p in weights.items()}
 (E/'comparison.json').write_text(json.dumps({'results':results,'adopted':False,'scope':plan['scope']},indent=2),encoding='utf-8')
 print(json.dumps({k:r['counts'] for k,r in results.items()}))
