"""Apply the prewritten absolute/relative criteria after the candidate is frozen."""
import json
import os
from pathlib import Path
from PIL import Image,ImageDraw

from run_experiment import BASE,box_iou,fixed_metrics,sha256,verify_snapshot,write_json
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work'/'yolo-config')

RUN=BASE/'experiments'/'lvis_board_ccby_trial'

def framework(model,data,folder):
    result=model.val(data=str(data),split='test',imgsz=640,batch=8,device=0,workers=0,plots=False,verbose=False,project=str(RUN),name=folder)
    return {'map50':float(result.box.map50),'map50_95':float(result.box.map)}

def negatives_and_sheets(model,records,tag):
    predictions=[];negative_boxes=0;negative_images=0
    for row in records:
        result=model.predict(row['image'],conf=.5,imgsz=640,device=0,verbose=False)[0]
        boxes=result.boxes.xyxy.cpu().tolist();scores=result.boxes.conf.cpu().tolist()
        if row['boxes']==0:negative_images+=1;negative_boxes+=len(boxes)
        predictions.append({'image':row['image'],'boxes':boxes,'confidences':scores,'ground_truth_boxes':row['boxes']})
    folder=RUN/'comparison_sheets';folder.mkdir(exist_ok=True)
    for start in range(0,len(records),12):
        sheet=Image.new('RGB',(1200,1040),'white')
        for k,(row,prediction) in enumerate(zip(records[start:start+12],predictions[start:start+12])):
            im=Image.open(row['image']).convert('RGB');w,h=im.size;draw=ImageDraw.Draw(im)
            for line in Path(row['label']).read_text(encoding='utf-8-sig').splitlines():
                _,x,y,bw,bh=map(float,line.split());draw.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline='lime',width=3)
            for box,confidence in zip(prediction['boxes'],prediction['confidences']):
                draw.rectangle(box,outline='red',width=3);draw.text((box[0]+2,box[1]+2),f'{confidence:.2f}',fill='red')
            im.thumbnail((400,230));x=k%3*400;y=k//3*260;sheet.paste(im,(x,y))
            ImageDraw.Draw(sheet).text((x+3,y+232),f"{Path(row['image']).stem} GT:{row['boxes']} P:{len(prediction['boxes'])}",fill='black')
        sheet.save(folder/f'{tag}_{start//12+1}.jpg')
    return {'negative_images':negative_images,'false_boxes':negative_boxes,
        'false_boxes_per_negative_image':negative_boxes/negative_images if negative_images else 0.,'predictions':predictions}

def main():
    target=BASE/'lvis_board_comparison.json'
    if target.exists():raise RuntimeError('Comparison already completed; preserve the first holdout evaluation')
    state=json.loads((RUN/'report.json').read_text(encoding='utf-8'))
    if state['status']!='completed' or 'test' not in state:raise RuntimeError('Candidate run has not completed')
    locked=json.loads((RUN/'locked_plan.json').read_text(encoding='utf-8'))
    rule_file=Path(locked['plan']['external_relative_decision_rule'])
    if sha256(rule_file)!=locked['plan']['external_relative_decision_rule_sha256']:raise RuntimeError('External decision criteria changed')
    rule=json.loads(rule_file.read_text(encoding='utf-8'));limits=rule['all_required']
    if sha256(rule['reference_weights'])!=locked['initial_weights_sha256']:raise RuntimeError('Reference weights changed')
    fresh_audit=json.loads((RUN/'data_audit.json').read_text(encoding='utf-8'))
    fresh=[r for r in fresh_audit['records'] if r['split']=='test']
    old_audit=json.loads((BASE/'experiments'/'history_audit_handoff16_final'/'data_audit.json').read_text(encoding='utf-8'))
    old=[r for r in old_audit['records'] if r['split']=='test']
    verify_snapshot({'records':fresh+old})
    from ultralytics import YOLO
    reference=YOLO(rule['reference_weights']);candidate=YOLO(state['candidate_weights'])
    old_fresh_framework=framework(reference,rule['fresh_data'],'fresh_reference_evaluation')
    old_fresh_fixed=fixed_metrics(reference,fresh,['cutting_board'],.5,640,0,.5)
    new_fresh_fixed=state['test']['fixed'];new_fresh_framework=state['test']['framework']
    if new_fresh_fixed['confidence']!=.5:raise RuntimeError('Candidate confidence disagrees with prewritten comparison')
    negatives_ref=negatives_and_sheets(reference,fresh,'reference')
    negatives_new=negatives_and_sheets(candidate,fresh,'candidate')
    historical={}
    for name,model in [('reference',reference),('candidate',candidate)]:
        historical[name]={'framework':framework(model,BASE/'board_dataset'/'data.yaml',f'historical_{name}_regression'),
            'fixed':fixed_metrics(model,old,['cutting_board'],.5,640,0,.5)}
    manifest=json.loads((BASE/'board_lvis_ccby'/'manifest.json').read_text(encoding='utf-8'))
    origins={r['image_file']:r['image']['coco_url'].split('/')[-2] for r in manifest['records'] if r['source']=='LVIS_v1'}
    strata={}
    for origin in ['train2017','val2017']:
        subset=[r for r in fresh if origins[r['image']]==origin]
        strata[origin]={'images':len(subset),'positive_images':sum(r['boxes']>0 for r in subset),
            'reference':fixed_metrics(reference,subset,['cutting_board'],.5,640,0,.5),
            'candidate':fixed_metrics(candidate,subset,['cutting_board'],.5,640,0,.5)}
    before=old_fresh_fixed['micro'];after=new_fresh_fixed['micro']
    checks=[]
    def minimum(name,value,required):checks.append({'criterion':name,'observed':value,'minimum':required,'passed':value>=required})
    def maximum(name,value,required):checks.append({'criterion':name,'observed':value,'maximum':required,'passed':value<=required})
    minimum('fresh precision',after['precision'],limits['fresh_min_precision'])
    minimum('fresh recall',after['recall'],limits['fresh_min_recall'])
    minimum('fresh map50',new_fresh_framework['map50'],limits['fresh_min_map50'])
    minimum('fresh F1 improvement',after['f1']-before['f1'],limits['fresh_min_f1_improvement'])
    maximum('fresh precision drop',before['precision']-after['precision'],limits['fresh_max_precision_drop'])
    maximum('fresh recall drop',before['recall']-after['recall'],limits['fresh_max_recall_drop'])
    maximum('fresh extra false boxes per negative image',negatives_new['false_boxes_per_negative_image']-negatives_ref['false_boxes_per_negative_image'],limits['fresh_max_extra_false_boxes_per_negative_image'])
    maximum('historical map50 drop',historical['reference']['framework']['map50']-historical['candidate']['framework']['map50'],limits['historical_max_map50_drop'])
    maximum('historical fixed recall drop',historical['reference']['fixed']['micro']['recall']-historical['candidate']['fixed']['micro']['recall'],limits['historical_max_fixed_recall_drop'])
    pipeline_decision=json.loads((RUN/'decision.json').read_text(encoding='utf-8'))
    passed=all(c['passed'] for c in checks) and pipeline_decision['criteria_met']
    payload={'rule_file':str(rule_file),'rule_sha256':sha256(rule_file),'all_predefined_criteria_passed':passed,
        'reference_weights':rule['reference_weights'],'candidate_weights':state['candidate_weights'],
        'fresh_reference':{'framework':old_fresh_framework,'fixed':old_fresh_fixed,'negative_probe':negatives_ref},
        'fresh_candidate':{'framework':new_fresh_framework,'fixed':new_fresh_fixed,'negative_probe':negatives_new},
        'historical_regression':historical,'checks':checks,'pipeline_absolute_criteria':pipeline_decision['checks'],
        'pretraining_source_strata':strata,
        'strata_note':'Exploratory provenance breakdown, not a new acceptance criterion. Generic YOLO weights were pretrained on COCO; some LVIS holdout source photos are COCO train2017 images.',
        'registered_model_changed':False,
        'limitations':['Public LVIS/COCO photos new to board-label development, not canteen footage.','Only 38 test images, 15 positive and 23 explicitly negative.','23 source images come from COCO train2017; generic YOLO pretraining may have seen them for other classes. No globally unseen-image claim.','Historical test used only as a predefined regression guard, not independent performance evidence.','Photo/source filtering does not prove independent households or scenes.']}
    write_json(target,payload)
    print(json.dumps({'all_predefined_criteria_passed':passed,'fresh_reference':before,'fresh_candidate':after,'checks':checks},indent=2),flush=True)

if __name__=='__main__':main()
