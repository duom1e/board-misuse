"""Class-aware one-to-one IoU matching for bounded food-group development ROIs."""
import os,json
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'food_group_boxes_v1'
def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return x/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-x)
def main():
 from ultralytics import YOLO
 data=json.loads((E/'manifest.json').read_text(encoding='utf-8'));rows=data['records'];result=[]
 models={'original':B/'training_runs/foodseg_2cls/weights/best.pt','balanced128':B/'food_balanced_detector_v1/training/weights/last.pt','replay512':B/'food_replay512_detector_v1/training/weights/last.pt','positive3':B/'food_replay512_positive3_detector_v1/training/weights/last.pt'}
 for name,weights in models.items():
  model=YOLO(str(weights))
  for r in rows:
   p=model.predict(r['image'],imgsz=640,rect=False,conf=.25,device='cpu',verbose=False)[0];pred=[dict(box=q.xyxy[0].tolist(),category=model.names[int(q.cls[0])],confidence=float(q.conf[0])) for q in p.boxes];used=set();tp=0
   for gt in r['boxes_xyxy']:
    matches=[(iou(gt,q['box']),i) for i,q in enumerate(pred) if i not in used and q['category']==r['category'] and iou(gt,q['box'])>=.5]
    if matches:used.add(max(matches)[1]);tp+=1
   result.append(dict(model=name,id=r['id'],source=r['source_group'],tp=tp,fn=len(r['boxes_xyxy'])-tp,unmatched_predictions=len(pred)-tp,predictions=pred))
 (E/'model_comparison.json').write_text(json.dumps(dict(rows=result,conf=.25,imgsz=640,rect=False,match_iou=.5,training_performed=False),ensure_ascii=False,indent=2),encoding='utf-8')
 text=['# 多来源食材框标注与检测对照','','新增6张限定ROI、15个可见食材组框：猪肉5、鸡肉6、蔬菜4。框由助手逐图核对，非专家、非全帧穷尽实例标注。四张猪肉/辣椒ROI列作训练候选，两张鸡肉ROI保留来源隔离开发验证；这些原视频此前已消费，不能称独立盲测。尚未用这6张ROI训练。','','这次做真实类别+IoU>=.5的一对一框匹配，不能与之前裁剪“有候选”混为同一指标。所有模型同conf=.25、640、rect=False。','', '|模型|匹配/15|未匹配组框|未匹配预测|','|---|---:|---:|---:|']
 for name in models:
  rs=[r for r in result if r['model']==name];text.append(f'|{name}|{sum(r["tp"] for r in rs)}|{sum(r["fn"] for r in rs)}|{sum(r["unmatched_predictions"] for r in rs)}|')
 text+=['','未匹配预测不能全部称假食材：组框粒度、局部预测和同组拆分均可能造成不匹配。ROI内种子/极小碎屑不单独标实例；身体接触未核实。下一轮可使用4训练候选ROI补充512回放，鸡肉源必须继续隔离，并保持旧验证集。','', '原图/ROI/标注SHA及来源已保存在manifest，全部叠框在review.jpg；数据和模型留本机。']
 (E/'报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8');print('\n'.join(text[6:13]))
if __name__=='__main__':main()
