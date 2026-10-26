"""One preplanned larger-capacity candidate, evaluated on development validation only."""
import json,os,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work/yolo-config')
os.environ['YOLO_AUTOINSTALL']='False';os.environ['YOLO_OFFLINE']='True'
from experiment_data import audit_dataset,freeze_dataset,verify_snapshot,sha256
from run_experiment import fixed_metrics,write_json
from ultralytics import YOLO

RUN=BASE/'experiments/board_capacity_s_validation'
PLAN=BASE/'board_capacity_s_plan.json'

def main():
    plan=json.loads(PLAN.read_text(encoding='utf-8-sig'))
    RUN.mkdir(parents=True,exist_ok=False)
    write_json(RUN/'locked_plan.json',{'plan':plan,'plan_sha256':sha256(PLAN),'initial_weights_sha256':sha256(plan['initial_weights'])})
    audit=audit_dataset(plan['data'])
    write_json(RUN/'data_audit.json',audit)
    if not audit['passed']:raise RuntimeError('Dataset audit failed')
    data=freeze_dataset(audit,RUN/'frozen_dataset')
    # Training YAML omits test entirely: no test prediction or evaluation in this experiment.
    import yaml
    config=yaml.safe_load(data.read_text(encoding='utf-8-sig'));config.pop('test',None)
    validation_data=RUN/'train_val_only.yaml'
    validation_data.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    write_json(RUN/'status.json',{'state':'training','started':time.time()})
    model=YOLO(plan['initial_weights'])
    model.train(data=str(validation_data),project=str(RUN),name='training',exist_ok=False,plots=False,verbose=False,**plan['training'])
    verify_snapshot(audit)
    write_json(RUN/'status.json',{'state':'validation_comparison'})
    records=[r for r in audit['records'] if r['split']=='val']
    outputs={}
    for key,weights in [('reference',plan['reference_weights']),('candidate',str(RUN/'training/weights/best.pt'))]:
        m=YOLO(weights)
        fixed=fixed_metrics(m,records,['cutting_board'],.5,640,0,.5)
        framework=m.val(data=str(validation_data),split='val',imgsz=640,device=0,workers=0,plots=False,project=str(RUN),name=key+'_val',verbose=False)
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
