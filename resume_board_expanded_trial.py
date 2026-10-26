"""One preplanned larger-capacity candidate, evaluated on development validation only."""
import json,os,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work/yolo-config')
os.environ['YOLO_AUTOINSTALL']='False';os.environ['YOLO_OFFLINE']='True'
from experiment_data import audit_dataset,freeze_dataset,verify_snapshot,sha256
from run_experiment import fixed_metrics,write_json
from ultralytics import YOLO

RUN=BASE/'experiments/board_expanded_s_reviewed'
PLAN=BASE/'board_expanded_s_plan.json'

def main():
    plan=json.loads(PLAN.read_text(encoding='utf-8-sig'))
    locked=json.loads((RUN/'locked_plan.json').read_text(encoding='utf-8'))
    if sha256(PLAN)!=locked['plan_sha256']: raise RuntimeError('Plan changed; cannot resume')
    audit=json.loads((RUN/'data_audit.json').read_text(encoding='utf-8'))
    verify_snapshot(audit)
    checkpoint=RUN/'training/weights/last.pt'
    import torch
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    epoch=state.get('epoch')
    if epoch is None or not 0<=epoch<49 or state.get('optimizer') is None: raise RuntimeError('Invalid resume checkpoint')
    write_json(RUN/'recovery.json',{'reason':'CUDA unknown error; fresh process GPU tensor smoke passed','checkpoint_epoch_zero_based':epoch,'checkpoint_sha256':sha256(checkpoint),'same_frozen_plan':True,'note':'Resume restores optimizer and epoch; data-loader random sequence after interruption may differ.'})
    del state
    write_json(RUN/'status.json',{'state':'resuming_training','checkpoint_epoch_zero_based':epoch})
    model=YOLO(str(checkpoint))
    model.train(resume=True,device=0)
    verify_snapshot(audit)
    write_json(RUN/'status.json',{'state':'validation_comparison'})
    records=[r for r in audit['records'] if r['split']=='val']
    outputs={}
    for key,weights in [('reference',plan['reference_weights']),('candidate',str(RUN/'training/weights/best.pt'))]:
        m=YOLO(weights)
        fixed=fixed_metrics(m,records,['cutting_board'],.5,640,0,.5)
        framework=m.val(data=str(RUN/'train_val_only.yaml'),split='val',imgsz=640,device=0,workers=0,plots=False,project=str(RUN),name=key+'_val',verbose=False)
        outputs[key]={'weights':weights,'sha256':sha256(weights),'fixed':fixed,'map50':float(framework.box.map50),'map50_95':float(framework.box.map)}
    ref=outputs['reference']['fixed']['micro'];cand=outputs['candidate']['fixed']['micro']
    criteria=plan['validation_selection_criteria']
    decision={'precision_minimum':cand['precision']>=criteria['minimum_precision'],
              'recall_minimum':cand['recall']>=criteria['minimum_recall'],
              'recall_gain':cand['recall']-ref['recall']>=criteria['minimum_recall_gain'],
              'f1_gain':cand['f1']-ref['f1']>=criteria['minimum_f1_gain']}
    result={'scope':'Reused development validation, not independent test or deployment accuracy',
            'results':outputs,'criteria':decision,'all_passed':all(decision.values()),
            'test_evaluated':False,'registry_changed':False,
            'limitations':plan['limitations']}
    write_json(RUN/'comparison.json',result)
    write_json(RUN/'status.json',{'state':'completed','all_passed':result['all_passed']})
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if RUN.exists():write_json(RUN/'status.json',{'state':'failed','error':str(exc)})
        traceback.print_exc();raise
