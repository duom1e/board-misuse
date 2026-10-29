"""Full-frame vs manual-ROI vs fixed grid, conditional GT coverage only."""
import os,json
from pathlib import Path
from PIL import Image,ImageDraw
from evaluate_food_group_boxes import iou
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'food_scale_diagnosis_v1'
def nms(ps):
 keep=[]
 for q in sorted(ps,key=lambda x:-x['confidence']):
  if all(q['category']!=r['category'] or iou(q['box'],r['box'])<.5 for r in keep):keep.append(q)
 return keep
def main():
 E.mkdir(exist_ok=False)
 from ultralytics import YOLO
 model=YOLO(str(B/'food_yellow_chicken_adaptation_v1/training/weights/last.pt'))
 data=json.loads((B/'food_group_boxes_v1/manifest.json').read_text(encoding='utf-8'))['records'];rows=[]
 for r in data:
  if r['split']!='source_held_development':continue
  full=Image.open(r['source_frame']).convert('RGB');w,h=full.size;rx,ry,_,_=r['roi'];gt=[[a+rx,b+ry,c+rx,d+ry] for a,b,c,d in r['boxes_xyxy']]
  tw=min(w,800);th=min(h,600);tiles=list(dict.fromkeys((x,y,x+tw,y+th) for y in [0,h-th] for x in [0,w-tw]))
  for mode,boxes in [('full',[(0,0,w,h)]),('manual_roi',[tuple(r['roi'])]),('fixed_grid',tiles)]:
   ps=[]
   for tile in boxes:
    pred=model.predict(full.crop(tile),imgsz=640,rect=False,conf=.25,device='cpu',verbose=False)[0]
    for q in pred.boxes:
     a,b,c,d=q.xyxy[0].tolist();ps.append(dict(box=[a+tile[0],b+tile[1],c+tile[0],d+tile[1]],category=model.names[int(q.cls[0])],confidence=float(q.conf[0])))
   ps=nms(ps);counts={}
   for label in [False,True]:
    used=set();tp=0
    for g in gt:
     matches=[(iou(g,q['box']),j) for j,q in enumerate(ps) if j not in used and (not label or q['category']=='meat') and iou(g,q['box'])>=.5]
     if matches:used.add(max(matches)[1]);tp+=1
    counts['category_correct' if label else 'geometry']=tp
   row=dict(id=r['id'],mode=mode,gt_count=len(gt),predictions=ps,tiles=boxes,**counts);rows.append(row)
   im=full.copy();draw=ImageDraw.Draw(im)
   for g in gt:draw.rectangle(g,outline='lime',width=3)
   for q in ps:draw.rectangle(q['box'],outline='orange',width=3);draw.text((q['box'][0],q['box'][1]),q['category'],fill='orange')
   im.save(E/f'{r["id"]}_{mode}.jpg')
 (E/'results.json').write_text(json.dumps(dict(rows=rows,conf=.25,imgsz=640,rect=False,nms_iou=.5,scope='Two consumed sources frames; conditional matching to6 GT within bounded ROIs. Outside ROI food not exhaustively labeled, no full-frame precision. Manual ROI is oracle localization; fixed grid independent of GT.'),ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# 整图与分块尺度诊断','','固定黄皮候选模型、conf=.25、640/rect=False；三种输入都映射回原图，按同6组框IoU>=.5一对一匹配。固定网格为800x600四角窗，重叠框按同类别IoU>=.5去重，不调阈值。','', '|输入|类别无关匹配/6|肉类正确匹配/6|候选总数|','|---|---:|---:|---:|']
 for mode in ['full','manual_roi','fixed_grid']:
  rs=[r for r in rows if r['mode']==mode];lines.append(f'|{mode}|{sum(r["geometry"] for r in rs)}|{sum(r["category_correct"] for r in rs)}|{sum(len(r["predictions"]) for r in rs)}|')
 lines+=['','人工ROI是已知位置辅助，不能当自动定位能力。整图ROI之外食材没有穷尽标注，不能计算整图精确率或把额外框都称误报。固定网格没有使用GT位置，但本批已消费开发样本，不是独立测试。分块会增加计算与重复框，类别混淆仍需单独评估。未训练、未切换默认，未接自动视频分块。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines[4:9]))
if __name__=='__main__':main()
