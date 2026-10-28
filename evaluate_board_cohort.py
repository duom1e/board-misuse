"""Same-input grouped diagnostic across frozen historical detector candidates."""
import os,json,hashlib
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'board_cohort_v1'
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 from PIL import Image
 from ultralytics import YOLO
 E.mkdir(exist_ok=True);assert not (E/'locked_plan.json').exists(),'Inspect prior run; do not restart'
 plan=json.loads((B/'multisource_negative_trial_v1/locked_plan.json').read_text(encoding='utf-8'));cases=plan['new_train']+[dict(image=x['image'],box=x['box'],group=x['id']) for x in plan['heldout']];assert len(cases)==30
 seen=set()
 for i,r in enumerate(cases):
  sha=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest();assert sha not in seen;seen.add(sha);r.update(index=i,sha256=sha)
  if not r['box']:r['stratum']='no_visible_board'
  elif r['group']=='BV1dHckzwEBA' and Path(r['image']).stem in ('300','390'):r['stratum']='small_edge_partial'
  elif r['group']=='BV1hK411u73L':r['stratum']='board_fills_frame'
  elif r['group']=='BV15J411D718':r['stratum']='round_board_on_wood_table'
  elif r['group']=='BV1dHckzwEBA':r['stratum']='hand_occluded_round_board'
  else:r['stratum']='rectangular_board'
 models={'default':json.loads((B/'active_models.json').read_text(encoding='utf-8'))['board']['weights'],'ordinary':str(B/'kitchen_cutting_trial_v1/train/weights/last.pt'),'conservative':str(B/'kitchen_cutting_trial_conservative_v1/train/weights/last.pt'),'multisource':str(B/'multisource_board_trial_v1/train/weights/last.pt'),'negative_augmented':str(B/'multisource_negative_trial_v1/train/weights/last.pt')}
 save('locked_plan.json',dict(cases=cases,models=models,conf=.25,iou=.5,rect=False,imgsz=640,scope='All consumed development cases. Model-specific video train exposure, no blind pooled accuracy.'))
 rows=[]
 exposure={'default':set(),'ordinary':{'BV1dHckzwEBA'},'conservative':{'BV1dHckzwEBA'},'multisource':{r['group'] for r in plan['new_train'] if r['group']!='BV1dC7H6PEoG'},'negative_augmented':{r['group'] for r in plan['new_train']}}
 for name,w in models.items():
  m=YOLO(w)
  for r in cases:
   out=[]
   for q in m.predict(r['image'],imgsz=640,rect=False,conf=.25,verbose=False)[0].boxes:
    z=q.xyxy[0].tolist();iou=0
    if r['box']:
     a,b,c,d=r['box'];x,y,v,k=z;it=max(0,min(c,v)-max(a,x))*max(0,min(d,k)-max(b,y));iou=it/((c-a)*(d-b)+(v-x)*(k-y)-it)
    out.append(dict(box=z,confidence=float(q.conf[0]),iou=iou))
   match=bool(r['box']) and any(q['iou']>=.5 for q in out);rows.append(dict(model=name,case=r['index'],group=r['group'],stratum=r['stratum'],train_source_exposed=r['group'] in exposure[name],positive=bool(r['box']),matched=match,unmatched_boxes=len(out)-int(match),false_negative=bool(r['box']) and not match,negative_false_frame=not r['box'] and bool(out),predictions=out))
  save('partial_results.json',dict(rows=rows));print(name,'completed',flush=True)
 groups=[]
 for name in models:
  for source in sorted({r['group'] for r in cases}):
   rs=[r for r in rows if r['model']==name and r['group']==source];groups.append(dict(model=name,source=source,train_exposed=rs[0]['train_source_exposed'],positives=sum(r['positive'] for r in rs),matches=sum(r['matched'] for r in rs),negative_frames=sum(not r['positive'] for r in rs),negative_false_frames=sum(r['negative_false_frame'] for r in rs),extra_boxes=sum(r['unmatched_boxes'] for r in rs)))
 save('results.json',dict(rows=rows,groups=groups,training_performed=False,adopted=False))
 lines=['# 案板模型统一分组诊断','','30张已消费图，7视频来源，18正例12负例；同输入评5模型。按模型列训练来源暴露，不能混作独立测试准确率。','', '|模型|来源|训练暴露|匹配/正例|负例误帧/负例|额外框|','|---|---|---|---|---|---|']
 lines += [f"|{g['model']}|{g['source']}|{g['train_exposed']}|{g['matches']}/{g['positives']}|{g['negative_false_frames']}/{g['negative_frames']}|{g['extra_boxes']}|" for g in groups]
 lines+=['','阈值conf .25 / IoU .5，一GT最多匹配一框。额外框含低重叠框与重复框，不得只报告命中而忽略它们。来源暴露仅针对本项目新增视频，原默认预训练重叠未知。人工非专家标签，截断外接框宽松，类别/接触/错用未评。','', '下一批优先补未参与训练或调参的边缘截断板、手遮挡圆板、木纹桌面圆板，以及锅盘容器负例。每一类尽量覆盖不同拍摄来源；保留整来源，不随机拆相邻帧。现有30图固定开发回归，不再反复称新测试。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
