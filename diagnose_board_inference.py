"""Validation-only inference comparison. Does not retune on the 38-image test set."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

from run_experiment import BASE,box_iou,precision_recall,sha256,verify_snapshot,write_json
from experiment_data import read_boxes

RUN=BASE/'experiments'/'board_inference_diagnostic'

def predictions(model,image,imgsz):
    result=model.predict(image,imgsz=imgsz,conf=.5,device=0,verbose=False)[0]
    return [(list(map(float,box)),float(score)) for box,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist())]

def tiled_predictions(model,image):
    width,height=image.size
    combined=predictions(model,image,640)
    tile_width=math.ceil(width*.625);tile_height=math.ceil(height*.625)
    for x in sorted({0,width-tile_width}):
        for y in sorted({0,height-tile_height}):
            crop=image.crop((x,y,x+tile_width,y+tile_height))
            for box,score in predictions(model,crop,640):
                left,top,right,bottom=box
                if min(left,top,tile_width-right,tile_height-bottom)<.02*min(tile_width,tile_height):continue
                if (right-left)*(bottom-top)>.10*width*height:continue
                combined.append(([left+x,top+y,right+x,bottom+y],score))
    kept=[]
    for box,score in sorted(combined,key=lambda x:-x[1]):
        if all(box_iou(box,previous)<.5 for previous,_ in kept):kept.append((box,score))
    return kept

def evaluate(model,records,strategy):
    counts={'tp':0,'fp':0,'fn':0};small_total=0;small_hit=0;misses=[];rows=[]
    for row in records:
        with Image.open(row['image']) as opened:image=opened.convert('RGB')
        width,height=image.size
        if strategy=='full640_tiles':pred=tiled_predictions(model,image)
        else:pred=predictions(model,image,int(strategy.removeprefix('full')))
        truth=read_boxes(Path(row['label']),['cutting_board'])
        boxes=[[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height] for _,x,y,w,h in truth]
        short=[min(b[2]-b[0],b[3]-b[1])*640/max(width,height) for b in boxes]
        small_total+=sum(s<32 for s in short);used=set();false=[]
        for box,score in sorted(pred,key=lambda p:-p[1]):
            candidate=max([(box_iou(box,b),i) for i,b in enumerate(boxes) if i not in used],default=(0,None))
            if candidate[0]>=.5 and candidate[1] is not None:
                counts['tp']+=1;used.add(candidate[1]);small_hit+=short[candidate[1]]<32
            else:counts['fp']+=1;false.append({'box':box,'confidence':score})
        counts['fn']+=len(boxes)-len(used)
        for i,b in enumerate(boxes):
            if i not in used:misses.append({'image':row['image'],'gt_box':b,'short_side_at_640':short[i],
                'best_predicted_iou':max([box_iou(b,p) for p,_ in pred] or [0])})
        rows.append({'image':row['image'],'gt_boxes':len(boxes),'predictions':[{'box':b,'confidence':c} for b,c in pred],'unmatched_predictions':false})
    return {'overall':precision_recall(counts),'small_targets':{'gt':small_total,'matched':small_hit,
        'recall':small_hit/small_total if small_total else None,'evidence_sufficient':small_total>=5},
        'missed_ground_truth':misses,'images':rows}

def main():
    if RUN.exists():raise RuntimeError('Preserve existing diagnostic; do not overwrite or keep sweeping validation settings')
    registry=json.loads((BASE/'active_models.json').read_text(encoding='utf-8-sig'))
    weights=registry['board_lvis']['weights']
    audit=json.loads((BASE/'experiments'/'lvis_board_ccby_trial'/'data_audit.json').read_text(encoding='utf-8'))
    records=[r for r in audit['records'] if r['split']=='val']
    verify_snapshot({'records':records})
    RUN.mkdir(parents=True)
    plan={'scope':'Development validation only. Not an independent test or small-target generalization study.',
        'images':len(records),'weights':weights,'weights_sha256':sha256(weights),
        'confidence':.5,'match_iou':.5,'strategies':['full640','full960','full1280','full640_tiles'],
        'tile_policy':'Four 62.5%-size corner tiles; 2% edge margin; retain tile boxes <=10% original image area; NMS IoU0.5.',
        'all_criteria':{'minimum_precision':.70,'maximum_precision_drop':.03,'minimum_recall_gain':.05},
        'small_target_definition':'normalized short side below32 after proportional scaling to640; not COCO area definition',
        'small_target_evidence_note':'Only one such labeled target here. Small-target conclusions are underpowered.',
        'registration_will_not_change':True}
    write_json(RUN/'locked_diagnostic_plan.json',plan)
    from ultralytics import YOLO
    model=YOLO(weights);results={}
    for strategy in plan['strategies']:
        results[strategy]=evaluate(model,records,strategy)
        write_json(RUN/f'{strategy}.json',results[strategy])
        print(strategy,json.dumps(results[strategy]['overall']),flush=True)
    baseline=results['full640']['overall'];decisions=[]
    for strategy,r in results.items():
        current=r['overall']
        passed=(current['precision']>=.70 and current['precision']>=baseline['precision']-.03 and current['recall']>=baseline['recall']+.05)
        decisions.append({'strategy':strategy,'meets_predefined_validation_criteria':passed,
            'precision_change':current['precision']-baseline['precision'],'recall_change':current['recall']-baseline['recall']})
    payload={'plan':plan,'summary':{k:{'overall':v['overall'],'small_targets':v['small_targets']} for k,v in results.items()},
        'decisions':decisions,'default_or_registry_changed':False,
        'limitations':['Development validation already used for model selection.','Only one small target; no small-target reliability conclusion.','No test-set threshold or resolution selection.']}
    write_json(RUN/'summary.json',payload)
    lines=['# 案板推理方式诊断','', '仅使用57张开发验证图、28个案板框，固定confidence0.5和匹配IoU0.5；不使用38张测试图选择参数。','',
        '| 方式 | TP | FP | FN | 精确率 | 召回率 | F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for key,r in payload['summary'].items():
        m=r['overall'];lines.append(f"| {key} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.3f} |")
    lines+=['','预写条件：P至少0.70，相比640的P最多下降0.03，R至少增加0.05。','',
        '满足全部条件的方式：'+', '.join(d['strategy'] for d in decisions if d['meets_predefined_validation_criteria'])+'。','',
        '短边在640输入下小于32的GT只有1个，不能用其成绩证明小案板泛化。','',
        '放大不增加源图片的信息量，切片可能放大局部木纹并产生额外误报。本次不更换注册表或默认推理方式。','',
        '逐图预测和漏检框见各策略JSON，计划见locked_diagnostic_plan.json。']
    (RUN/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(RUN/'report.md',flush=True)

if __name__=='__main__':main()
