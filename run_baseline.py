"""Initial one-class cutting-board baseline, not misuse recognition."""
from pathlib import Path
import json
import os

BASE = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(BASE.parent / 'work' / 'yolo-config')

def main():
    import torch
    from ultralytics import YOLO
    os.chdir(BASE)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    data = BASE / 'board_dataset' / 'data.yaml'
    model = YOLO('yolo11n.pt')
    model.train(data=str(data),epochs=20,imgsz=640,batch=8,device=0,workers=0,
                project=str(BASE/'training_runs'),name='board_baseline',exist_ok=False,
                seed=42,deterministic=True,plots=True,cache=False,amp=False)
    run = Path(model.trainer.save_dir)
    best = YOLO(str(run/'weights'/'best.pt'))
    metrics = best.val(data=str(data),split='test',imgsz=640,batch=8,device=0,workers=0,
                       project=str(BASE/'training_runs'),name='board_test',plots=True)
    result={'task':'cutting-board detection only; not board misuse',
            'epochs':20,'imgsz':640,'batch':8,'seed':42,
            'weights':str(run/'weights'/'best.pt'),
            'test_metrics':{k:float(v) for k,v in metrics.results_dict.items()},
            'test_images':len((BASE/'board_dataset'/'test.txt').read_text().splitlines())}
    (BASE/'baseline_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    best.predict(source=str(BASE/'board_dataset'/'test.txt'),imgsz=640,device=0,
                 conf=0.25,save=True,project=str(BASE/'training_runs'),name='board_predictions')
    print('BASELINE_COMPLETE',json.dumps(result),flush=True)

if __name__ == '__main__':
    main()
