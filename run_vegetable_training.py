from pathlib import Path
import json
import os
BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work'/'yolo-config')

def main():
    from ultralytics import YOLO
    if (BASE/'training_runs'/'vegetable_baseline'/'weights'/'best.pt').exists():
        raise RuntimeError('Existing trained experiment found. Choose a new experiment name before rerunning.')
    data=BASE/'vegetable_dataset'/'data.yaml'
    model=YOLO(str(BASE/'yolo11n.pt'))
    model.train(data=str(data),epochs=20,imgsz=640,batch=8,device=0,workers=0,
        project=str(BASE/'training_runs'),name='vegetable_baseline',seed=42,
        deterministic=True,amp=False,plots=True,exist_ok=True)
    run=Path(model.trainer.save_dir)
    best=YOLO(str(run/'weights'/'best.pt'))
    metrics=best.val(data=str(data),split='test',imgsz=640,batch=8,device=0,workers=0,
        project=str(BASE/'training_runs'),name='vegetable_test',plots=True)
    result={'task':'vegetable detection; no meat detection or raw/cooked recognition',
        'epochs':20,'weights':str(run/'weights'/'best.pt'),
        'test_metrics':{k:float(v) for k,v in metrics.results_dict.items()},
        'test_images':len((BASE/'vegetable_dataset'/'test.txt').read_text().splitlines())}
    (BASE/'vegetable_metrics.json').write_text(json.dumps(result,indent=2))
    best.predict(source=str(BASE/'vegetable_dataset'/'test.txt'),save=True,conf=.5,device=0,
        project=str(BASE/'training_runs'),name='vegetable_predictions')
    print('VEGETABLE_TRAINING_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__': main()
