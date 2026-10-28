import json
from pathlib import Path
from PIL import Image,ImageDraw
from ultralytics import YOLO
from run_canteen_food_detector import evaluate
from run_food_mixed_selection import operational
import report_redveg_contrast_trial as retention
B=Path(__file__).resolve().parent;OLD=B/'experiments/redveg_contrast_frozen_backbone';NEW=B/'experiments/redveg_contrast_unfrozen_control'
if __name__=='__main__':
 a=json.loads((OLD/'decision.json').read_text());b=json.loads((NEW/'decision.json').read_text());rows=json.loads((B/'redveg_meat_contrast_v1/manifest.json').read_text(encoding='utf-8'))['records']
 p=NEW/'training_fit_diagnostic.json'
 if p.exists():fit=json.loads(p.read_text())
 else:
  r=evaluate(YOLO(b['selected']['weights']),rows);fit={'counts':r['counts'],'classes':operational(r)['classes'],'rows':r['rows'],'scope':'Training fit only, never used for checkpoint selection'};p.write_text(json.dumps(fit,indent=2),encoding='utf-8')
 p=NEW/'old_per_class_diagnostic.json'
 if p.exists():classes=json.loads(p.read_text())
 else:
  retention.E=NEW;classes=retention.class_metrics(b['selected']['weights'],'old_classes_after');p.write_text(json.dumps(classes,indent=2),encoding='utf-8')
 gt={r['image']:r for r in json.loads((B/'bilibili_prep_manual_v1/manifest.json').read_text(encoding='utf-8'))['records']};frames=[1195,1465,1855,2005];sheet=Image.new('RGB',(1160,1260),'white')
 for j,(name,d) in enumerate([('freeze10',a),('freeze0',b)]):
  for i,f in enumerate(frames):
   r=next(r for r in d['bilibili20']['rows'] if gt[r['image']]['frame']==f);im=Image.open(r['image']).convert('RGB');dr=ImageDraw.Draw(im)
   for box in gt[r['image']]['boxes_xyxy']:dr.rectangle(box,outline='lime',width=2)
   for p in r['predictions']:
    dr.rectangle(p['xyxy'],outline='red',width=3);dr.text((p['xyxy'][0],max(0,p['xyxy'][1]-13)),f'{["meat","vegetable"][p["class"]]} {p["confidence"]:.2f}',fill='red')
   x=j*580;y=i*315;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+290),f'{name} f{f} TP{r["tp"]} FP{r["fp"]} FN{r["fn"]}',fill='black')
 sheet.save(NEW/'freeze_comparison.jpg')
 lines=[]
 for name,d in [('冻结前10模块',a),('解除冻结',b)]:
  r=d['bilibili20_summary'];lines.append(f'|{name}|{Path(d["selected"]["weights"]).name}|{r["tp"]}|{r["fp"]}|{r["fn"]}|{r["meat_predictions"]}|{d["old_after"]["map50"]:.4f}|{d["operational_gate"]}|')
 oldfit=json.loads((OLD/'new_training_fit_diagnostic.json').read_text())
 text='''# 骨干冻结策略受控比较

2026-10-27。此轮实跑解除冻结的30轮训练，原freeze10结果保留作对照。

## 事前方案与审计

初始权重、580张不同训练图/612曝光条目、三份输入列表内容、来源标注SHA、学习率.0001、AdamW、batch8、640像素、seed42、30轮、增强参数、候选保存与选择规则均相同；只把freeze从10改为0，输出目录另设。comparison_audit.json核对三份列表SHA相同。采用同一随机种子是受控比较，不是多种子统计证据。

仍在15张双机位开发验证图上选模型，固定conf=.5、IoU=.5，操作门槛和旧64图回归门槛不变。选定后一次评B站20张已暴露开发ROI，不用该结果换权重、阈值或追加训练。仅为定位问题，不自动上线。

## 结果

|方案|所选权重|B站TP|FP|FN|肉预测框|旧64图mAP50|操作门槛|
|---|---|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(lines)+f'''

解除冻结：旧域回归门槛{b['regression_gate']}，B站诊断门槛{b['bilibili20_gate']}。旧域原始参考mAP50为{b['old_before']['map50']:.4f}。肉AP50补充诊断为{classes['0']['AP50']:.4f}，菜AP50为{classes['1']['AP50']:.4f}，均未用于选择模型。

新16张训练图拟合（非测试）：freeze10 TP{oldfit['result']['counts']['tp']}/26，freeze0 TP{fit['counts']['tp']}/26。解除冻结肉类训练召回{fit['classes']['meat']['recall']:.4f}，菜类训练召回{fit['classes']['vegetable']['recall']:.4f}。仅用于分清“训练样本是否学会”与跨来源诊断，不能当成泛化准确率。两方案各自按同一验证政策选择权重，所选轮次可能不同；比较的是完整选择流程的结果，不是固定同一epoch的参数效应。

## 边界

不替换默认、不注册、不发布候选。实际注册表未改变：{b['registry_unchanged']}。本轮没有证明真实错用识别可用。标注为助手粗区域，旧验证帧与训练邻近，类别与新视频来源仍有关联；B站20图已经用于开发。FP还包括IoU不足的局部框。无预测不能算空板合规。不是VLM微调。

目录experiments/redveg_contrast_unfrozen_control保存预锁计划、审计、训练、全部候选验证、decision、训练拟合、各类回归和freeze_comparison.jpg。原freeze10实验未覆盖。原始媒体和权重保留本机，无GitHub上传或外发消息。
'''
 (B/'骨干冻结策略对照实验报告.md').write_text(text,encoding='utf-8')
 print(json.dumps({'unfrozen':b['bilibili20_summary'],'old_map50':b['old_after']['map50'],'gates':[b['operational_gate'],b['regression_gate'],b['bilibili20_gate']],'train_fit':fit['counts']}))
