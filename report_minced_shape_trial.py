import json
from pathlib import Path
from PIL import Image,ImageDraw
from ultralytics import YOLO
from run_canteen_food_detector import evaluate
from run_food_mixed_selection import operational
B=Path(__file__).resolve().parent;E=B/'experiments/minced_shape_adaptation_v1_reviewed'
if __name__=='__main__':
 d=json.loads((E/'decision.json').read_text());reserved=json.loads((E/'reserved_source_decision.json').read_text());prev=json.loads((B/'experiments/redveg_contrast_unfrozen_control/decision.json').read_text());source=json.loads((B/'minced_shape_annotation_v1/manifest.json').read_text(encoding='utf-8'))
 p=E/'extra_training_fit.json'
 if p.exists():fit=json.loads(p.read_text())
 else:
  r=evaluate(YOLO(d['selected']['weights']),[r for r in source['records'] if r['intended_role']=='training_candidate']);fit={'result':r,'classes':operational(r)['classes'],'scope':'Training fit only, not validation accuracy'};p.write_text(json.dumps(fit,indent=2),encoding='utf-8')
 lines=[]
 for name,r in [('上一轮解除冻结',prev),('本轮补细碎形态',d)]:
  c=r['bilibili20_summary'];lines.append(f'|{name}|{c["tp"]}|{c["fp"]}|{c["fn"]}|{c["meat_predictions"]}|{r["old_after"]["map50"]:.4f}|')
 res='操作或旧域门槛未通过，保留来源没有运行模型预测，继续保留。'
 if reserved['released']:
  rr=[]
  for name,r in reserved['results'].items():
   s=r['summary'];passed=s['recall']>=.5 and (s['precision'] or 0)>=.75 and s['meat_predictions']==0
   rr.append(f'- {name}：TP{s["tp"]}/FP{s["fp"]}/FN{s["fn"]}，肉预测{s["meat_predictions"]}，预定菜类诊断门槛通过={passed}。')
  res='\n'.join(rr)
  gt={r['image']:r for r in source['records']};sheet=Image.new('RGB',(1280,5*385),'white')
  for j,name in enumerate(['previous','candidate']):
   for i,r in enumerate(reserved['results'][name]['result']['rows']):
    im=Image.open(r['image']).convert('RGB');dr=ImageDraw.Draw(im)
    for box in gt[r['image']]['boxes_xyxy']:dr.rectangle(box,outline='lime',width=3)
    for p in r['predictions']:
     dr.rectangle(p['xyxy'],outline='red',width=4);dr.text((p['xyxy'][0],max(0,p['xyxy'][1]-14)),f'{["meat","vegetable"][p["class"]]} {p["confidence"]:.2f}',fill='red')
    im.thumbnail((640,360));x=j*640;y=i*385;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+363),f'{name} {r["seconds"]}s TP{r["tp"]} FP{r["fp"]} FN{r["fn"]}',fill='black')
  sheet.save(E/'reserved_source_comparison.jpg')
 text=f'''# 细碎形态补样混训实验

2026-10-27。真实完成30轮训练，默认模型没有替换。

## 方案与数据

保持上一轮解除冻结设置：原foodseg初始化、30轮、freeze0、AdamW lr.0001、640、batch8、seed42。旧512图+原后厨52图+早先16图曝光3次，再加新11图曝光3次，共591独特图/645训练条目。新11图来自同一案板的5肉6菜，包含肉末和细碎红椒，不能把重复曝光算新图片。新增数据使更新量增加，不是等计算量消融。

同样15张双机位开发图选权重，固定conf=.5、IoU=.5，操作/旧64图回归门槛不变。选中{Path(d['selected']['weights']).name}。操作门槛={d['operational_gate']}，旧域回归门槛={d['regression_gate']}。

保留的BV14K4y1x7LF整来源不进训练、不选模型。事前排除44秒高遮挡图后，5图6个蔬菜区域。仅操作/旧域门槛都过才释放：{reserved['released']}。通过后同时比较上一轮候选与新候选一次，不按该分数换权重。菜类诊断预定R>=.5、P>=.75且肉预测0；无肉/空板样本，不能代表完整系统。

## 已暴露B站开发片段

|模型|TP|FP|FN|肉预测|旧64图mAP50|
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(lines)+f'''

18个蔬菜区域+2无明显块状图。此片段早已用于开发，不能当独立最终测试。匹配是粗组框IoU，FP包含类别错和局部定位不匹配。

## 保留来源首次模型诊断

{res}

保留图经过人工标注检查，但此前未跑模型；若本轮释放，今后即为已消费开发证据，不再叫新鲜测试。5帧仍来自同一个短视频、帧相关强；不能推广为跨厨房准确率。2菜组框在32秒同图，其余各1框。

## 训练拟合及边界

新11张训练图的补充拟合为TP{fit['result']['counts']['tp']}/FP{fit['result']['counts']['fp']}/FN{fit['result']['counts']['fn']}，只是训练拟合，不是泛化指标。候选不采用、不注册，注册表SHA未变={d['registry_unchanged']}。没有生熟、真实接触或自动错用能力声明，也不是VLM微调。

启动前首次预检发现新裁图标签没有images/labels约定目录，断言中止，未发生该次训练。修正为将新11图和标签复制到实验内data/images/train及data/labels/train并核对SHA后，在新目录启动；原失败目录minced_shape_adaptation_v1保留，勿认作有效训练。实际结果目录为minced_shape_adaptation_v1_reviewed。

锁计划、审计、训练权重、全部候选验证、decision.json、reserved_source_decision.json和拟合诊断均在experiments/minced_shape_adaptation_v1_reviewed。标签单助手非专家；前景ROI外其他食材未穷尽标注。原图、视频、权重本机保存，未发布、未联系创作者。
'''
 (B/'细碎形态补样混训实验报告.md').write_text(text,encoding='utf-8')
 print(json.dumps({'selected':Path(d['selected']['weights']).name,'old_map50':d['old_after']['map50'],'bilibili20':d['bilibili20_summary'],'gates':[d['operational_gate'],d['regression_gate'],d['bilibili20_gate']],'reserved_released':reserved['released'],'reserved':{k:r['summary'] for k,r in reserved['results'].items()},'training_fit':fit['result']['counts']}))
