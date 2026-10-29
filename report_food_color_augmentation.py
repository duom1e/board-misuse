"""Evaluate new detector against the locked group-box cohort and old regression."""
import os,json,hashlib
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
from evaluate_food_group_boxes import iou
B=Path(__file__).resolve().parent;E=B/'food_color_augmentation_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 from ultralytics import YOLO
 data=read(B/'food_group_boxes_v1/manifest.json')['records'];plan=read(E/'locked_plan.json');oldplan=read(B/'food_replay512_detector_v1/locked_plan.json')
 assert plan['old_validation']==oldplan['old_validation'] and plan['old_training']==oldplan['old_training']
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 train=(E/'train.txt').read_text().splitlines();assert len(train)==548
 assert not set(map(sha,train))&set(map(sha,plan['old_validation']))
 assert sha(B/'active_models.json')==plan['registry_sha256']
 model=YOLO(str(E/'training/weights/last.pt'));out=[]
 for r in data:
  p=model.predict(r['image'],conf=.25,imgsz=640,rect=False,device='cpu',verbose=False)[0];pred=[dict(box=q.xyxy[0].tolist(),category=model.names[int(q.cls[0])],confidence=float(q.conf[0])) for q in p.boxes];used=set();tp=0
  for gt in r['boxes_xyxy']:
   matches=[(iou(gt,q['box']),i) for i,q in enumerate(pred) if i not in used and q['category']==r['category'] and iou(gt,q['box'])>=.5]
   if matches:used.add(max(matches)[1]);tp+=1
  out.append(dict(id=r['id'],split=r['split'],tp=tp,fn=len(r['boxes_xyxy'])-tp,unmatched_predictions=len(pred)-tp,predictions=pred))
 previous=read(B/'food_group_boxes_v1/model_comparison.json')['rows'];lines=['# 颜色亮度增强检测训练对照','','512旧图+16负例+11已有正例+4猪肉辣椒ROI，另加3张另一来源鸡肉ROI，再加2张黄皮鸡ROI，共548图，15轮AdamW .0001/640/batch8/seed42固定last；从原foodseg起训，未微调大模型。鸡肉2张6组框未加入训练。所有来源此前已消费，非盲测。','', '|模型|训练候选框匹配/9|鸡肉留出框匹配/6|未匹配预测|','|---|---:|---:|---:|']
 held={r['id'] for r in data if r['split']=='source_held_development'}
 for name,rows in [('原模型',[r for r in previous if r['model']=='original']),('512回放',[r for r in previous if r['model']=='replay512']),('颜色增强',out)]:lines.append(f'|{name}|{sum(r["tp"] for r in rows if r["id"] not in held)}|{sum(r["tp"] for r in rows if r["id"] in held)}|{sum(r["unmatched_predictions"] for r in rows)}|')
 before=read(E/'old_before.json');after=read(E/'old_after.json');probe=read(E/'after.json');bg=[r for r in probe if r['reference']=='background'];food=[r for r in probe if r['reference']!='background']
 lines+=['',f'旧64图 mAP50 {before["map50"]:.4f} → {after["map50"]:.4f}；mAP50-95 {before["map50_95"]:.4f} → {after["map50_95"]:.4f}。',f'固定裁剪诊断：背景有候选 {sum(bool(r["predictions"]) for r in bg)}/23；食材有候选 {sum(bool(r["predictions"]) for r in food)}/19；出现参考类别 {sum(any(q["category"]==r["reference"] for q in r["predictions"]) for r in food)}/19。','','类别+IoU>=.5一对一匹配；新增训练框表现不能当验证准确率。未匹配预测可能是局部框或组框粒度差异，不全部称假食材。默认模型未替换；保留最后轮权重和逐图原始结果。训练结束，无后台。']
 (E/'group_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');(E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines[4:12]))
if __name__=='__main__':main()
