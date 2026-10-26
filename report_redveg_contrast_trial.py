import json
from pathlib import Path
from PIL import Image,ImageDraw
import run_food_mixed_selection as selection
from ultralytics import YOLO
B=Path(__file__).resolve().parent;E=B/'experiments/redveg_contrast_frozen_backbone'
def class_metrics(weights,name):
 r=YOLO(str(weights)).val(data=str(E/'data/old_regression.yaml'),split='val',imgsz=640,conf=.001,iou=.7,device=0,batch=8,workers=0,plots=False,project=str(E),name=name,verbose=False)
 return {str(int(c)):{'class':r.names[int(c)],'AP50':float(r.box.ap50[i]),'AP50_95':float(r.box.ap[i])} for i,c in enumerate(r.box.ap_class_index)}
if __name__=='__main__':
 d=json.loads((E/'decision.json').read_text(encoding='utf-8'));plan=json.loads((E/'locked_plan.json').read_text(encoding='utf-8'))
 p=E/'old_per_class_diagnostic.json'
 if p.exists():classes=json.loads(p.read_text())
 else:
  classes={'before':class_metrics(plan['initial_weights'],'old_classes_before'),'after':class_metrics(d['selected']['weights'],'old_classes_after')};p.write_text(json.dumps(classes,indent=2),encoding='utf-8')
 prior=json.loads((B/'experiments/bilibili_food_frozen_probe/comparison.json').read_text(encoding='utf-8'))['results'];pd=json.loads((B/'experiments/bilibili_food_frozen_probe/classification_diagnostic.json').read_text(encoding='utf-8'))
 gt={r['image']:r for r in json.loads((B/'bilibili_prep_manual_v1/manifest.json').read_text(encoding='utf-8'))['records']}
 frames=[1195,1465,1855,2005];sheet=Image.new('RGB',(1160,1260),'white')
 for j,(name,result) in enumerate([('previous_mixed_candidate',prior['previous_mixed_candidate']),('contrast_trial',d['bilibili20'])]):
  for i,f in enumerate(frames):
   r=next(r for r in result['rows'] if gt[r['image']]['frame']==f);im=Image.open(r['image']).convert('RGB');dr=ImageDraw.Draw(im)
   for box in gt[r['image']]['boxes_xyxy']:dr.rectangle(box,outline='lime',width=2)
   for box in r['predictions']:
    dr.rectangle(box['xyxy'],outline='red',width=3);dr.text((box['xyxy'][0],max(0,box['xyxy'][1]-13)),f'{["meat","vegetable"][box["class"]]} {box["confidence"]:.2f}',fill='red')
   sheet.paste(im,(j*580,i*315));ImageDraw.Draw(sheet).text((j*580+3,i*315+290),f'{name} f{f} TP{r["tp"]} FP{r["fp"]} FN{r["fn"]}',fill='black')
 sheet.save(E/'bilibili_prediction_review.jpg')
 rows=[]
 for name,r in prior.items():
  c=r['counts'];z=pd[name];rows.append(f'|{name}|{c["tp"]}|{c["fp"]}|{c["fn"]}|{z["predicted_classes"].get("meat",0)}|{z["predicted_classes"].get("vegetable",0)}|')
 s=d['bilibili20_summary'];rows.append(f'|本轮候选|{s["tp"]}|{s["fp"]}|{s["fn"]}|{s["meat_predictions"]}|{s["vegetable_predictions"]}|')
 cr=[]
 for k in ['0','1']:
  a=classes['before'][k];z=classes['after'][k];cr.append(f'|{a["class"]}|{a["AP50"]:.4f}|{z["AP50"]:.4f}|{a["AP50_95"]:.4f}|{z["AP50_95"]:.4f}|')
 text=f'''# 红椒混淆对照混训实验

2026-10-26。本轮确实完成训练，未替换默认模型。

## 预先固定的方案

从原始foodseg权重出发，冻结前10个模块，训练其余部分30轮，640像素、batch8、AdamW lr.0001、seed42。数据为512张旧食材训练图+52张原双视角后厨图各一次+16张新红椒/肉类图各曝光3次，共580张独特图/612训练条目。新图重复不算新增独特图。扩大旧覆盖、调整曝光、冻结和学习率同时改变，不能把效果归因某一个因素。

只用原15张双机位开发验证选择候选，每10轮保存及best/last参与，固定conf=.5、IoU=.5。总P>=.75、肉R>=2/3、菜R>=.5、各机位肉/菜R>=.5、四张无明显块状验证图误框0；达标优先后按macroF1等固定顺序排名。选中{Path(d['selected']['weights']).name}，操作门槛通过：{d['operational_gate']}。

旧域仍同64张旧验证图，mAP50最多降.05，实际{d['old_before']['map50']:.4f}→{d['old_after']['map50']:.4f}，门槛通过：{d['regression_gate']}。

原B站20张ROI不进训练或选模型，仅选定权重后评一次，专门检查新训练来源之外的已暴露开发问题。预定诊断目标TP>=6/18、肉预测框<=4且P>=.5；通过：{d['bilibili20_gate']}。这个目标只是开发改善门槛，远不等于可交付准确率。

## 同一B站开发片段的结果

|模型|TP|FP|FN|预测肉框|预测菜框|
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

18个蔬菜粗区域GT，另2张无明显块状食材；没有真实肉阳性。减少肉框只有同时增加正确蔬菜定位才有意义，不能靠全部不报来宣布解决。FP包括错误类别和与粗区域IoU不足的局部定位框，不能全解释成凭空检测。

## 旧食材各类保留情况（补充描述性检查）

|类别|原AP50|候选AP50|原AP50-95|候选AP50-95|
|---|---:|---:|---:|---:|
'''+ '\n'.join(cr)+f'''

各类AP为选定权重后补充诊断，未用于选模型，不是预设的额外选型门槛。曲线AP与conf=.5固定阈值精确率不同。64图样本有限，不能证明肉类识别全面保留。

## 边界与资产

没有改阈值、没有看B站结果换权重或追加训练。候选不采用、不注册，active_models.json哈希未改：{d['registry_unchanged']}。原双视角验证是同操作近邻图，新对照每类一个视频，类别与场景有关联；B站20图虽没进训练，已经被多次看过做开发，非独立最终测试。标签单助手非专家粗区域，台面/刀具/字幕/遮挡仍可能干扰。当前不是VLM训练，不能判生熟、接触、清洁或自动错用。

实验目录experiments/redveg_contrast_frozen_backbone：locked_plan.json、data_audit.json、全部候选验证、decision.json、逐图B站诊断、旧域指标/补充各类AP、bilibili_prediction_review.jpg及权重。脚本run_redveg_contrast_trial.py、report_redveg_contrast_trial.py。新来源图保持出处和复核记录，创作者再利用许可未建立，不发布原图或权重，不发送消息。
'''
 (B/'红椒混淆对照混训实验报告.md').write_text(text,encoding='utf-8')
 print(json.dumps({'selected':Path(d['selected']['weights']).name,'operational_gate':d['operational_gate'],'regression_gate':d['regression_gate'],'bilibili20_gate':d['bilibili20_gate'],'bilibili20':s,'old_per_class':classes},ensure_ascii=False))
