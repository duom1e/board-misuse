"""Correct known validation label omissions, then compare at identical resolution.

Original test set is deliberately not used in this development experiment.
"""
import csv,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work'/'yolo-config')

def prepare():
    source=BASE/'vegetable_dataset';dest=BASE/'vegetable_dataset_v2'
    if (dest/'data.yaml').exists():return dest/'data.yaml'
    audit=json.loads((BASE/'vegetable_label_audit.json').read_text())
    changes=[]
    for split in ['train','validation']:
        (dest/'images'/split).mkdir(parents=True,exist_ok=True)
        (dest/'labels'/split).mkdir(parents=True,exist_ok=True)
        originals=[Path(line) for line in (source/f'{split}.txt').read_text().splitlines() if line.strip()]
        for image in originals:
            target=dest/'images'/split/image.name
            if not target.exists():os.link(image,target)
            (dest/'labels'/split/f'{image.stem}.txt').write_text((source/'labels'/split/f'{image.stem}.txt').read_text())
        for r in audit[split]['subclass_boxes_without_parent_iou_0_5']:
            x1,y1,x2,y2=r['box'];label=dest/'labels'/split/f"{r['image_id']}.txt"
            with label.open('a') as f:f.write(f'0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}\n')
            changes.append(r)
        (dest/f'{split}.txt').write_text('\n'.join((dest/'images'/split/image.name).as_posix() for image in originals)+'\n')
    (dest/'label_changes.json').write_text(json.dumps({'changes':changes,'method':'Append independently annotated specific vegetable boxes with parent IoU <0.5; manually screened candidate images; not an exhaustive reannotation.'},indent=2))
    source_manifest=json.loads((source/'manifest.json').read_text())
    records=[r for r in source_manifest['images'] if r['split'] in ['train','validation']]
    ids={r['image_id'] for r in records}
    with (source/'attribution.csv').open(encoding='utf-8-sig') as f:
        attribution=[r for r in csv.DictReader(f) if r['ImageID'] in ids]
    with (dest/'attribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(attribution[0]));writer.writeheader();writer.writerows(attribution)
    (dest/'manifest.json').write_text(json.dumps({'source_manifest':'../vegetable_dataset/manifest.json',
        'images':records,'annotation_changes':'label_changes.json',
        'annotation_source':'Open Images; Google LLC CC BY 4.0; images retain per-image attribution'},indent=2))
    (dest/'data.yaml').write_text(f'path: {dest.as_posix()}\ntrain: train.txt\nval: validation.txt\nnames:\n  0: vegetable\n')
    return dest/'data.yaml'

def main():
    from ultralytics import YOLO
    if (BASE/'vegetable_refinement_metrics.json').exists():raise RuntimeError('Refinement already completed; preserve the previous experiment.')
    data=prepare()
    old_path=Path(json.loads((BASE/'vegetable_metrics.json').read_text())['weights'])
    baseline=YOLO(str(old_path)).val(data=str(data),imgsz=768,batch=8,device=0,workers=0,
        project=str(BASE/'training_runs'),name='vegetable_v2_reference',plots=True)
    reference={k:float(v) for k,v in baseline.results_dict.items()}
    (BASE/'vegetable_v2_reference.json').write_text(json.dumps({'weights':str(old_path),'split':'validation','imgsz':768,'metrics':reference},indent=2))
    model=YOLO(str(old_path))
    model.train(data=str(data),epochs=25,imgsz=768,batch=8,device=0,workers=0,
        optimizer='AdamW',lr0=.0002,lrf=.1,warmup_epochs=1,freeze=10,
        mosaic=.25,scale=.2,close_mosaic=5,amp=False,seed=42,deterministic=True,
        project=str(BASE/'training_runs'),name='vegetable_refined',plots=True)
    run=Path(model.trainer.save_dir);new_path=run/'weights'/'best.pt'
    candidate=YOLO(str(new_path)).val(data=str(data),imgsz=768,batch=8,device=0,workers=0,
        project=str(BASE/'training_runs'),name='vegetable_v2_candidate',plots=True)
    metrics={k:float(v) for k,v in candidate.results_dict.items()}
    improved=metrics['metrics/mAP50-95(B)']>reference['metrics/mAP50-95(B)']
    result={'evaluation_scope':'Development validation set, corrected labels, 60 images. No test evaluation performed.',
        'imgsz':768,'extra_training_epochs':25,'reference_weights':str(old_path),'candidate_weights':str(new_path),
        'reference_validation_metrics':reference,'candidate_validation_metrics':metrics,
        'selection_metric':'metrics/mAP50-95(B)','candidate_selected':improved,
        'limitations':['Only 60 validation images; used for checkpoint and model selection.','Known omissions corrected; labels are not exhaustive.','No estimate of real-kitchen misuse accuracy.']}
    (BASE/'vegetable_refinement_metrics.json').write_text(json.dumps(result,indent=2))
    if improved:
        registry={'vegetable':{'weights':str(new_path),'imgsz':768,'status':'experimental; target domain unvalidated',
            'selection_record':'vegetable_refinement_metrics.json'}}
        (BASE/'active_models.json').write_text(json.dumps(registry,indent=2))
    print('REFINEMENT_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
