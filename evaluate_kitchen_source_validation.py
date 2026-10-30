import os,json,hashlib
from pathlib import Path
from evaluate_food_group_boxes import iou
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'kitchen_source_validation_v1'
MODELS={'original':B/'training_runs/foodseg_2cls/weights/best.pt',**{n:B/f/'training/weights/last.pt' for n,f in [('bamboo','food_bamboo_source_adaptation_v1'),('carrot','food_carrot_balance_v1'),('cleaning960','food_cleaning_negative960_v1'),('multiveg960','food_multiveg960_v1')]}}
def main():
 from ultralytics import YOLO
 m=E/'manifest.json';data=json.loads(m.read_text(encoding='utf-8'));rows=data['records'];assert all(r['review_status']=='assistant_overlay_reviewed_nonexpert' for r in rows)
 plan=dict(manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),models={k:dict(weights=str(w),sha256=hashlib.sha256(w.read_bytes()).hexdigest()) for k,w in MODELS.items()},imgsz=640,rect=False,conf=.25,iou_match=.5);(E/'locked_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8');out=[]
 for name,w in MODELS.items():
  model=YOLO(str(w))
  for r in rows:
   assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==r['sha256']
   p=model.predict(r['image'],conf=.25,imgsz=640,rect=False,device='cpu',verbose=False)[0];ps=[dict(box=q.xyxy[0].tolist(),category=model.names[int(q.cls[0])],confidence=float(q.conf[0])) for q in p.boxes];g=r['boxes_xyxy'][0]
   out.append(dict(model=name,id=r['id'],reference=r['category'],geometry_match=any(iou(g,q['box'])>=.5 for q in ps),class_match=any(iou(g,q['box'])>=.5 and q['category']==r['category'] for q in ps),predictions=ps))
 (E/'results.json').write_text(json.dumps(dict(rows=out,training_performed=False,adopted=False),ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# 新增厨房来源的固定验收对照','','4个人工限定ROI参考组框，1个历史已查看、未纳入近期增量检测训练的视频来源。3肉含鱼头、1菜；参考在本次模型比较前锁定，助手看图标注非专家。人工ROI降低定位难度，鱼头/皮脂/食材组框为粗参考，不是完整实例GT；容器外食材未穷尽标注，不计算全图精确率。','', '|模型|几何匹配/4|类别正确匹配/4|候选数|','|---|---:|---:|---:|']
 for name in MODELS:
  rs=[r for r in out if r['model']==name];lines.append(f'|{name}|{sum(r["geometry_match"] for r in rs)}|{sum(r["class_match"] for r in rs)}|{sum(len(r["predictions"]) for r in rs)}|')
 lines+=['','所有模型conf=.25、640/rect=False、IoU>=.5，来源与裁剪SHA已核验。4条相关画面不能代表泛化准确率，也不是4个独立厨房。视频历史曾用于查看/其他诊断，预训练是否重合未知，不称盲测。','', '这批正式加入固定开发验收，不能转训练后仍称留出。两张姜肉混合参考预先排除；没有看模型结果后改标签。未训练、未切换默认。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines[4:11]))
if __name__=='__main__':main()
