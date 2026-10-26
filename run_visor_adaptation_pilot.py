import json,os,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work/yolo-config');os.environ['YOLO_AUTOINSTALL']='False';os.environ['YOLO_OFFLINE']='True'
from experiment_data import audit_dataset,freeze_dataset,verify_snapshot,sha256
from run_experiment import fixed_metrics,write_json
from ultralytics import YOLO
import yaml
RUN=BASE/'experiments/visor_adaptation_pilot';PLAN=BASE/'visor_adaptation_pilot_plan.json'

def main():
    plan=json.loads(PLAN.read_text(encoding='utf-8-sig'));RUN.mkdir(parents=True,exist_ok=False)
    write_json(RUN/'locked_plan.json',{'plan':plan,'plan_sha256':sha256(PLAN),'initial_weights_sha256':sha256(plan['initial_weights'])})
    audit=audit_dataset(plan['data'],groups_file=plan['groups_file']);write_json(RUN/'data_audit.json',audit)
    if not audit['passed']:raise RuntimeError('Dataset audit failed')
    frozen=freeze_dataset(audit,RUN/'frozen_dataset');data=yaml.safe_load(frozen.read_text());data.pop('test',None)
    training_data=RUN/'train_val_only.yaml';training_data.write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    write_json(RUN/'status.json',{'state':'training','started':time.time()})
    YOLO(plan['initial_weights']).train(data=str(training_data),project=str(RUN),name='training',plots=False,verbose=False,**plan['training'])
    verify_snapshot(audit)
    write_json(RUN/'status.json',{'state':'validation_comparison'})
    weights={'reference':plan['initial_weights'],'candidate':str(RUN/'training/weights/best.pt')};models={k:YOLO(v) for k,v in weights.items()}
    result={'scope':'Participant-disjoint VISOR adaptation, positive-only val/test; local research, no deployment claim','weights':weights,'weight_sha256':{k:sha256(v) for k,v in weights.items()},'validation':{}}
    val=[r for r in audit['records'] if r['split']=='val']
    for key,model in models.items():result['validation'][key]=fixed_metrics(model,val,['cutting_board'],.5,640,0,.5)
    ref=result['validation']['reference']['micro'];cand=result['validation']['candidate']['micro'];g=plan['validation_gate']
    checks={'precision':cand['precision']>=g['minimum_precision'],'recall':cand['recall']>=g['minimum_recall'],'recall_gain':cand['recall']-ref['recall']>=g['minimum_recall_gain']}
    result['validation_checks']=checks;result['validation_all_passed']=all(checks.values());write_json(RUN/'validation_comparison.json',result)
    # Existing LVIS validation is regression only, not a new independent score.
    historical=json.loads((BASE/'experiments/board_expanded_s_reviewed/data_audit.json').read_text())
    historical_rows=[r for r in historical['records'] if r['split']=='val']
    result['historical_regression']={key:fixed_metrics(model,historical_rows,['cutting_board'],.5,640,0,.5) for key,model in models.items()}
    result['test_evaluated']=False
    if result['validation_all_passed']:
        verify_snapshot(audit);write_json(RUN/'status.json',{'state':'frozen_test_evaluation'})
        test=[r for r in audit['records'] if r['split']=='test']
        result['test']={key:fixed_metrics(model,test,['cutting_board'],.5,640,0,.5) for key,model in models.items()}
        result['test_evaluated']=True
        ref=result['test']['reference']['micro'];cand=result['test']['candidate']['micro'];g=plan['test_gate']
        result['test_checks']={'precision':cand['precision']>=g['minimum_precision'],'recall':cand['recall']>=g['minimum_recall'],'recall_gain':cand['recall']-ref['recall']>=g['minimum_recall_gain']}
        result['test_all_passed']=all(result['test_checks'].values())
    result['registry_changed']=False;result['limitations']=plan['limitations'];write_json(RUN/'comparison.json',result)
    write_json(RUN/'status.json',{'state':'completed','validation_all_passed':result['validation_all_passed'],'test_evaluated':result['test_evaluated']})
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if RUN.exists():write_json(RUN/'status.json',{'state':'failed','error':str(exc)})
        traceback.print_exc();raise
