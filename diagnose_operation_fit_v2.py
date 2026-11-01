"""Compare training-source fit and cross-source development separately."""
import os,json
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
from ultralytics import YOLO
from evaluate_food_group_boxes import iou
B=Path(__file__).resolve().parent;E=B/'operation_fit_audit_v2'
def main():
 E.mkdir(exist_ok=False)
 datasets=['meat_operation_replay_v1','pov_operation_replay_v1','chopping_chicken_replay_v1'];models=['food_meat_operation_training_v1','food_pov_operation_training_v1','food_chopping_chicken_training_v1','food_chopping_exposure5_training_v1'];rows=[]
 for folder in models:
  model=YOLO(str(B/folder/'training/weights/last.pt'))
  for ds in datasets:
   records=json.loads((B/ds/'manifest.json').read_text(encoding='utf-8'))['records'];geometry=correct=total=predcount=0
   for r in records:
    boxes=r['boxes'];category='vegetable' if ds=='pov_operation_replay_v1' else 'meat'
    p=model.predict(r['image'],conf=.25,imgsz=640,rect=False,device='cpu',verbose=False)[0];ps=[dict(box=q.xyxy[0].tolist(),category=model.names[int(q.cls[0])],confidence=float(q.conf[0])) for q in p.boxes];predcount+=len(ps);total+=len(boxes)
    counts=[]
    for class_required in [False,True]:
     used=set();n=0
     for gt in boxes:
      choices=[(iou(gt,q['box']),i) for i,q in enumerate(ps) if i not in used and (not class_required or q['category']==category) and iou(gt,q['box'])>=.5]
      if choices:used.add(max(choices)[1]);n+=1
     counts.append(n)
    geometry+=counts[0];correct+=counts[1]
   rows.append(dict(model=folder,dataset=ds,geometry=geometry,correct=correct,total=total,predictions=predcount))
 (E/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 lines=['# 操作样本拟合诊断','','这些都是训练用样本或后续加入的训练样本，不能当泛化准确率。固定640/.25/rect=False、类别及IoU>=.5一对一匹配。组框粒度会影响结果。','','|模型|样本组|几何匹配|类别正确匹配|候选数|','|---|---|---:|---:|---:|']
 for r in rows:lines.append(f'|{r["model"]}|{r["dataset"]}|{r["geometry"]}/{r["total"]}|{r["correct"]}/{r["total"]}|{r["predictions"]}|')
 lines+=['','这张表仅定位训练样本拟合和类别/几何差异。此前固定跨来源开发集结果另见第21次训练报告，不能用本表取代。默认未改，未进行新训练。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
if __name__=='__main__':main()
