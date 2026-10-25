"""Audit -> lock plan -> train -> validation calibration -> test -> decision/report.

Defaults to audit only. This runner never modifies active_models.json.
"""
import argparse
import copy
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work'/'yolo-config')
os.environ.setdefault('YOLO_AUTOINSTALL','False')
os.environ.setdefault('YOLO_OFFLINE','True')
os.environ.setdefault('PIP_CACHE_DIR',str(BASE.parent/'work'/'pip-cache'))

from experiment_data import absolute,audit_dataset,freeze_dataset,read_boxes,sha256,verify_snapshot

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def load_plan(path):
    path=Path(path).resolve();plan=json.loads(path.read_text(encoding='utf-8-sig'))
    required={'experiment_name','data','initial_weights','data_status','allow_training','training','evaluation','acceptance'}
    if not isinstance(plan,dict) or not required<=set(plan):raise ValueError(f'Plan requires {sorted(required)}')
    if not isinstance(plan['experiment_name'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',plan['experiment_name']):raise ValueError('Experiment name must be 1..64 ASCII letters/digits/_/-')
    if not isinstance(plan['allow_training'],bool):raise ValueError('allow_training must be a boolean')
    if plan['data_status'] not in {'fresh_holdout_declared','historical_development_only','synthetic_software_fixture'}:
        raise ValueError('Declare data_status explicitly; freshness is a caller declaration, not independently proven')
    for key in ['data','initial_weights','groups_file','near_duplicate_reviews']:
        if plan.get(key):plan[key]=str(absolute(plan[key],path.parent))
    if not Path(plan['initial_weights']).is_file():raise FileNotFoundError('Initial weights must already exist locally')
    if Path(plan['initial_weights']).suffix.lower()!='.pt':raise ValueError('This runner requires local .pt initial weights')
    permitted={'epochs','imgsz','batch','device','workers','amp','seed','deterministic','patience','optimizer','lr0','lrf','freeze','mosaic','scale','close_mosaic','warmup_epochs'}
    if not isinstance(plan['training'],dict) or set(plan['training'])-permitted:raise ValueError('Unsupported training keys; output paths and resume are managed by the runner')
    defaults={'epochs':30,'imgsz':640,'batch':8,'device':0,'workers':0,'amp':False,'seed':42,'deterministic':True,'patience':0}
    plan['training']={**defaults,**plan['training']}
    for key in ['epochs','imgsz','batch']:
        value=plan['training'][key]
        if not isinstance(value,int) or isinstance(value,bool) or value<=0:raise ValueError(f'{key} must be a positive integer')
    if plan['training']['workers']!=0:raise ValueError('Use workers=0 for this Windows entry point')
    evaluation=plan['evaluation']
    if not isinstance(evaluation,dict):raise ValueError('evaluation must be an object')
    if set(evaluation)-{'confidence','match_iou','calibration'}:raise ValueError('Unsupported evaluation keys')
    for key in ['confidence','match_iou']:
        value=evaluation.get(key,.5)
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not 0<value<=1:raise ValueError(f'Invalid evaluation {key}')
        evaluation[key]=value
    calibration=evaluation.get('calibration')
    if calibration is not None:
        if not isinstance(calibration,dict) or set(calibration)!={'confidence_grid','minimum_precision'}:raise ValueError('Calibration requires confidence_grid and minimum_precision')
        grid=calibration.get('confidence_grid')
        if not isinstance(grid,list) or not grid or any(not isinstance(c,(int,float)) or isinstance(c,bool) or not 0<c<=1 for c in grid):raise ValueError('Invalid calibration grid')
        minimum=calibration.get('minimum_precision')
        if not isinstance(minimum,(int,float)) or isinstance(minimum,bool) or not 0<=minimum<=1:raise ValueError('Calibration minimum_precision must be 0..1')
    if set(plan['acceptance'])!={'min_map50','min_map50_95','min_precision','min_recall'}:
        raise ValueError('Acceptance requires min_map50, min_map50_95, min_precision, min_recall')
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or not 0<=v<=1 for v in plan['acceptance'].values()):raise ValueError('Acceptance thresholds must be finite numbers in 0..1')
    return plan

def box_iou(left,right):
    intersection=max(0,min(left[2],right[2])-max(left[0],right[0]))*max(0,min(left[3],right[3])-max(left[1],right[1]))
    union=(left[2]-left[0])*(left[3]-left[1])+(right[2]-right[0])*(right[3]-right[1])-intersection
    return intersection/union if union>0 else 0

def precision_recall(counts):
    tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return {**counts,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}

def choose_validation_threshold(sweep,minimum_precision):
    valid=[r for r in sweep if r['micro']['precision']>=minimum_precision and r['micro']['recall']>0]
    if not valid:return None
    return max(valid,key=lambda r:(r['micro']['recall'],r['micro']['f1'],r['confidence']))['confidence']

def fixed_metrics(model,records,names,confidence,imgsz,device,match_iou):
    counts=[{'tp':0,'fp':0,'fn':0} for _ in names]
    for row in records:
        prediction=model.predict(row['image'],conf=confidence,imgsz=imgsz,device=device,verbose=False)[0]
        height,width=prediction.orig_shape
        boxes=read_boxes(Path(row['label']),names)
        truths=[(c,[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]) for c,x,y,w,h in boxes]
        used=set()
        detections=sorted(zip(prediction.boxes.xyxy.cpu().tolist(),prediction.boxes.cls.cpu().tolist(),prediction.boxes.conf.cpu().tolist()),key=lambda x:-x[2])
        for box,cls,_ in detections:
            cls=int(cls)
            candidates=[(box_iou(box,truth),i) for i,(category,truth) in enumerate(truths) if category==cls and i not in used]
            best=max(candidates,default=(0,None))
            if best[0]>=match_iou and best[1] is not None:counts[cls]['tp']+=1;used.add(best[1])
            else:counts[cls]['fp']+=1
        for i,(cls,_) in enumerate(truths):
            if i not in used:counts[cls]['fn']+=1
    return {'confidence':confidence,'match_iou':match_iou,'images':len(records),
        'micro':precision_recall({k:sum(c[k] for c in counts) for k in ['tp','fp','fn']}),
        'per_class':{name:precision_recall(counts[i]) for i,name in enumerate(names)},
        'matching':'Class-aware, confidence-ordered greedy one-to-one IoU match; fixed threshold, distinct from framework curve P/R.'}

def report(run,state):
    write_json(run/'report.json',state)
    lines=['# 实验运行报告','',f"状态：{state['status']}",f"数据用途声明：{state['plan']['data_status']}",'',
        '本报告不证明真实后厨可靠性，不改 active_models.json。fresh_holdout_declared 是调用者声明，仍需核查采集来源。','',
        f"冻结计划指纹：`{state['plan_sha256']}`。完整配置见 locked_plan.json。"]
    audit=state.get('audit')
    if audit:
        lines+=['','## 数据检查','',f"检查通过：{audit['passed']}。",'', '| 集合 | 图片 | 框 | 显式空标签 |','|---|---:|---:|---:|']
        for key,c in audit['counts'].items():lines.append(f"| {key} | {c['images']} | {c['boxes']} | {c['explicit_empty_labels']} |")
        lines+=['',f"跨集合完全重复：{len(audit['cross_split_exact_duplicates'])}；近似图片候选：{len(audit['cross_split_near_duplicates'])}；跨集合场景组：{len(audit['cross_split_groups'])}。",'']
        lines.extend(f'- {warning}' for warning in audit['warnings'])
        lines+=['','完整问题与文件指纹见 data_audit.json。']
    if state.get('test'):
        lines+=['','## 预定阈值下的测试结果','',f"confidence={state['test']['fixed']['confidence']}；匹配IoU={state['test']['fixed']['match_iou']}。",'',
            '| 指标 | 结果 |','|---|---:|']
        values={**state['test']['framework'],**{k:state['test']['fixed']['micro'][k] for k in ['precision','recall']}}
        for key,value in values.items():lines.append(f'| {key} | {value:.4f} |')
        lines+=['','mAP为框架评估，P/R为固定阈值逐框匹配。既有开发数据或合成数据上的结果不代表新的独立性能。']
    if state.get('decision'):lines+=['','## 判据结果','',f"决定：{state['decision']['outcome']}。",'', '逐条判据见 decision.json；结果不会自动替换默认模型。']
    if state.get('error'):lines+=['','错误：'+state['error']]
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def execute(plan_file,train=False,output=None):
    plan=load_plan(plan_file)
    if train and not plan['allow_training']:raise ValueError('This plan disables training; audit-only historical plans must not be retrained accidentally')
    run=Path(output).resolve() if output else BASE/'experiments'/f"{plan['experiment_name']}_{uuid.uuid4().hex[:10]}"
    if run.exists():raise FileExistsError(f'Output exists; no overwrite: {run}')
    run.mkdir(parents=True)
    lock={'plan':copy.deepcopy(plan),'source_plan':str(Path(plan_file).resolve()),
        'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'initial_weights_sha256':sha256(plan['initial_weights']),'mode':'train' if train else 'check'}
    write_json(run/'locked_plan.json',lock)
    digest=sha256(run/'locked_plan.json')
    state={'status':'checking_data','plan':plan,'plan_sha256':digest,'registration_changed':False}
    report(run,state)
    try:
        audit=audit_dataset(plan['data'],plan.get('near_duplicate_hamming',4),plan.get('groups_file'),plan.get('near_duplicate_reviews'))
        write_json(run/'data_audit.json',audit);state['audit']=audit
        if not audit['passed']:
            state['status']='data_check_blocked';report(run,state);return run,2
        if not train:
            state['status']='data_check_passed';report(run,state);return run,0
        data=freeze_dataset(audit,run/'frozen_dataset')
        verify_snapshot(audit)
        from ultralytics import YOLO
        state['status']='training';report(run,state)
        if sha256(plan['initial_weights'])!=lock['initial_weights_sha256']:raise RuntimeError('Initial weights changed after plan lock')
        model=YOLO(plan['initial_weights'])
        model.train(data=str(data),project=str(run),name='training',exist_ok=False,plots=True,cache=False,**plan['training'])
        weights=Path(model.trainer.save_dir)/'weights'/'best.pt'
        model=YOLO(str(weights))
        if [model.names[i] for i in sorted(model.names)]!=audit['names']:raise RuntimeError('Trained class order disagrees with frozen dataset')
        state['candidate_weights']=str(weights);state['status']='validating';report(run,state)
        verify_snapshot(audit)
        selected_confidence=plan['evaluation']['confidence']
        calibration=plan['evaluation'].get('calibration')
        if calibration:
            val_records=[r for r in audit['records'] if r['split']=='val']
            sweep=[fixed_metrics(model,val_records,audit['names'],c,plan['training']['imgsz'],plan['training']['device'],plan['evaluation']['match_iou']) for c in sorted(set(calibration['confidence_grid']))]
            choice=choose_validation_threshold(sweep,calibration['minimum_precision'])
            payload={'split':'val','minimum_precision':calibration['minimum_precision'],'sweep':sweep,'supported':choice is not None,
                'selection':'Max recall subject to minimum precision, then F1, then higher confidence. Test images not used.'}
            if choice is not None:selected_confidence=choice
            payload['selected_confidence']=choice
            write_json(run/'validation_calibration.json',payload)
            if choice is None:
                state['decision']={'outcome':'validation_calibration_failed','test_evaluated':False,'registration_changed':False}
                write_json(run/'decision.json',state['decision']);state['status']='completed';report(run,state);return run,0
        state['status']='testing';report(run,state)
        test=model.val(data=str(data),split='test',imgsz=plan['training']['imgsz'],batch=plan['training']['batch'],device=plan['training']['device'],workers=0,
            project=str(run),name='test_evaluation',plots=True,verbose=False)
        fixed=fixed_metrics(model,[r for r in audit['records'] if r['split']=='test'],audit['names'],selected_confidence,
            plan['training']['imgsz'],plan['training']['device'],plan['evaluation']['match_iou'])
        framework={'map50':float(test.box.map50),'map50_95':float(test.box.map)}
        state['test']={'framework':framework,'fixed':fixed}
        write_json(run/'test_metrics.json',state['test'])
        observed={**framework,'precision':fixed['micro']['precision'],'recall':fixed['micro']['recall']}
        checks=[{'criterion':key,'minimum':minimum,'observed':observed[key.removeprefix('min_')],
            'passed':observed[key.removeprefix('min_')]>=minimum} for key,minimum in plan['acceptance'].items()]
        passed=all(c['passed'] for c in checks)
        state['decision']={'outcome':('software_verification_only' if plan['data_status']=='synthetic_software_fixture' else ('candidate_requires_review' if passed else 'criteria_not_met')),
            'criteria_met':passed,'checks':checks,'test_evaluated':True,'selected_confidence':selected_confidence,
            'registration_changed':False,'data_status':plan['data_status'],'plan_sha256':digest}
        write_json(run/'decision.json',state['decision']);state['status']='completed';report(run,state)
        return run,0
    except BaseException as error:
        state['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        state['error']=f'{type(error).__name__}: {error}';report(run,state)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',help='Local JSON plan, with dataset and acceptance criteria')
    parser.add_argument('--train',action='store_true',help='Explicitly run training; otherwise only audit data')
    parser.add_argument('--output',help='New output directory; existing paths are never overwritten')
    args=parser.parse_args()
    try:
        run,code=execute(args.plan,args.train,args.output)
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as error:parser.exit(1,f'{type(error).__name__}: {error}\n')
    print(f'Report: {run/"report.md"}',flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
