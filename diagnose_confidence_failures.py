"""Locked confidence sweep: diagnostic only, no threshold selection or promotion."""
import os,json,hashlib
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
from ultralytics import YOLO
from evaluate_food_group_boxes import iou
B=Path(__file__).resolve().parent;E=B/'confidence_failure_audit_v1'
def main():
 E.mkdir(exist_ok=False)
 refs=json.loads((B/'kitchen_source_validation_v1/manifest.json').read_text(encoding='utf-8'))['records']
 crops=json.loads((B/'food_presence_patches_v3/reviewed_manifest.json').read_text(encoding='utf-8'))['records']
 models={'657_15':'food_meat_operation_training_v1','657_30':'food_meat_operation_30epochs_v1','660_15':'food_pov_operation_training_v1'};out=[];raw=[]
 for name,folder in models.items():
  w=B/folder/'training/weights/last.pt';model=YOLO(str(w));preds=[]
  for r in refs:
   p=model.predict(r['image'],conf=.01,imgsz=640,rect=False,device='cpu',verbose=False)[0];qs=[dict(box=q.xyxy[0].tolist(),category=model.names[int(q.cls[0])],confidence=float(q.conf[0])) for q in p.boxes];preds.append((r,qs));raw.append(dict(model=name,id=r['id'],predictions=qs))
  bg=[]
  for r in crops:
   if r['label']!='background':continue
   p=model.predict(r['crop'],conf=.01,imgsz=640,rect=False,device='cpu',verbose=False)[0];bg.append([float(q.conf[0]) for q in p.boxes])
  for threshold in [.01,.05,.1,.25]:
   geom=correct=0
   for r,qs in preds:
    ps=[q for q in qs if q['confidence']>=threshold];g=r['boxes_xyxy'][0];geom+=any(iou(g,q['box'])>=.5 for q in ps);correct+=any(iou(g,q['box'])>=.5 and q['category']==r['category'] for q in ps)
   out.append(dict(model=name,threshold=threshold,geometry=geom,correct=correct,background_frames=sum(any(c>=threshold for c in cs) for cs in bg)))
  (E/f'{name}_weight_sha.txt').write_text(hashlib.sha256(w.read_bytes()).hexdigest())
 (E/'results.json').write_text(json.dumps(dict(rows=out,raw_kitchen=raw),indent=2),encoding='utf-8')
 lines=['# 置信度与漏检诊断','','固定4张厨房ROI、23张背景裁剪。阈值在本诊断运行前固定为.01/.05/.1/.25；NMS和IoU匹配阈值不变。开发集反复使用，不据此选择上线阈值。低阈值预测只用于诊断，不计正式模型进步。','','|模型|置信度|几何匹配/4|类别匹配/4|背景误帧/23|','|---|---:|---:|---:|---:|']
 for r in out:lines.append(f'|{r["model"]}|{r["threshold"]}|{r["geometry"]}|{r["correct"]}|{r["background_frames"]}|')
 lines+=['','几何匹配与类别匹配分开；没有候选、定位不准、类别错均可能导致类别匹配失败。降低阈值增加候选也可能增加背景误检，不能宣称解决识别。默认模型及阈值均未修改，未训练大模型。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
if __name__=='__main__':main()
