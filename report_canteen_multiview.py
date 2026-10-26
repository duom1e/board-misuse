import json
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'experiments/canteen_food_multiview_operational'
if __name__=='__main__':
 d=json.loads((E/'decision.json').read_text(encoding='utf-8'));s=d['selected_for_development'];ref=json.loads((E/'previous_reference_validation.json').read_text(encoding='utf-8'));allc=json.loads((E/'checkpoint_validation.json').read_text(encoding='utf-8'));manifest=json.loads((B/'canteen_food_multiview_v1/manifest.json').read_text(encoding='utf-8'));gt={r['image']:r for r in manifest['records']}
 views=s['operational']['views'];lines=[]
 for name,r in [('上一轮候选',ref),('本轮候选',s)]:
  for v,c in r['operational']['views'].items():
   q=c['counts'];lines.append(f'|{name}|{v}|{q["tp"]}|{q["fp"]}|{q["fn"]}|{c["classes"]["meat"]["recall"]:.3f}|{c["classes"]["vegetable"]["recall"]:.3f}|')
 table=[]
 for r in allc:
  c=r['validation']['counts'];table.append(f'|{Path(r["weights"]).name}|{c["tp"]}|{c["fp"]}|{c["fn"]}|{r["operational"]["macro_f1"]:.3f}|{r["operational"]["feasible"]}|')
 rs=[r for r in s['validation']['rows'] if gt[r['image']]['camera']=='side'];sheet=Image.new('RGB',(1240,1840),'white')
 for i,r in enumerate(rs):
  im=Image.open(r['image']).convert('RGB');dr=ImageDraw.Draw(im)
  for box in gt[r['image']]['boxes_xyxy']:dr.rectangle(box,outline='lime',width=2)
  for p in r['predictions']:
   dr.rectangle(p['xyxy'],outline='red',width=3);dr.text((p['xyxy'][0],max(0,p['xyxy'][1]-12)),f'{["meat","vegetable"][p["class"]]} {p["confidence"]:.2f}',fill='red')
  x=i%2*620;y=i//2*460;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+435),f'{r["seconds"]}s TP{r["tp"]} FP{r["fp"]} FN{r["fn"]}',fill='black')
 sheet.save(E/'selected_side_validation.jpg')
 before=d['old_before'];after=d['old_after'];old8=d['old8_development']['counts'] if d['old8_evaluated'] else '门槛未过，未评估'
 text=f'''# 第三十五轮：侧面视角标注与双视角混训

2026-10-26。

## 方向与用户需要做的事

当前先解决“案板上肉/蔬菜的可见区域定位”，再结合现场案板用途规则做疑似错用提示。仍是YOLO检测训练加规则，不是大模型/VLM微调；单张图也不能可靠判断生熟或真实接触。用户现在无需操作，现有视频仍可继续做数据标注和开发验证。有机会时从项目方获取实际使用规则、另一次操作/另一个厨房视频有帮助，但不是本轮开始工作的前提，未向任何人发消息。

## 本轮实际完成

抽取16张侧面训练候选，逐张核对后保留15张：4肉、10菜、1无明显块状食材。剔除120秒手持肉在案板边缘的归属模糊图，没有把有肉的图当空板负样本。采用粗食材聚集区域框，可能包含遮挡物与同类碎片间隙，非逐实例/专家标注；刀面反光未标食材，案板外其他食品不在标注范围。全部15新叠图已查看。

真实图共75：52训练、15验证、8旧开发留出。旧8张侧面诊断图本轮转作开发验证选择，已明确不再是选择外评估。训练混入同样128张旧食材训练图，真实52张各曝光3次，共180张独特图/284训练条目。重复曝光不是新图片。训练40轮640/batch8/AdamW lr.0002/seed42，从原始foodseg出发，保存每10轮及best/last。与上轮更新数不同，非等计算量消融。

训练前写锁定计划：conf=.5、IoU=.5，总精确率≥.75、肉召回≥2/3、菜召回≥.5、四张无明显块状食材验证图误框0，另外每个机位的肉/菜召回各≥.5。满足门槛优先，再按macroF1等固定排序。旧域回归仍用同样64张旧验证图，mAP50降幅不能超过.05。没用旧8张留出选权重。

## 结果

选作开发候选：{Path(s['weights']).name}；操作及旧域回归门槛合并通过：{d['development_and_regression_gates_passed']}。

|模型|机位开发集|TP|FP|FN|肉召回|菜召回|
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'''

|候选权重|TP|FP|FN|macroF1|过操作门槛|
|---|---:|---:|---:|---:|---|
'''+ '\n'.join(table)+f'''

旧64图mAP50：{before['map50']:.4f}→{after['map50']:.4f}；mAP50-95：{before['map50_95']:.4f}→{after['map50_95']:.4f}。这是曲线指标，不等于固定阈值精确率。

本轮失败原因是旧域回归降幅超过预定.05门槛（实际下降{before['map50']-after['map50']:.4f}），不是两个机位的操作门槛。说明仅128张旧图混训尚未阻止适配过程中的旧能力退化。不能只挑侧面5/6报成功。下一步优先扩大旧训练覆盖、控制新域曝光并验证保留旧能力；仍需扩大遮挡/形态覆盖和独立操作评估。

旧8张开发留出（仅合并门槛过后才评）：{old8}。

## 结论边界

同一厨房、同一次操作的不同机位与近邻帧关联很强，属于场景适配开发，不能称独立跨厨房准确率。侧面现在参与训练与选模型，即使分数改善也不能称未知视角泛化。粗框IoU匹配对不等于精细定位完全正确；手、刀与案板外食物仍可能干扰。无检测不能判断空板、清洁、合规。

不修改默认注册表，哈希核对未改变：{d['registry_unchanged']}；不发布候选权重，不接入自动错用结论。下一步先按逐图失败原因补遮挡和形态，再引入真正不同操作/厨房的评估与现场板用途规则。

## 资产与检查

- canteen_side_training_samples、canteen_food_multiview_v1：源裁图、75图标签、清单及15张新标注复核图。
- experiments/canteen_food_multiview_operational：预锁计划、审计、全部权重逐图验证、上一轮候选同图参照、旧域回归、决策及selected_side_validation.jpg（绿GT、红预测）。
- prepare_canteen_multiview.py，run_food_mixed_selection.py现支持数据集/实验参数及逐机位门槛，默认旧行为保持；test_operational_selection.py五项检查通过。
- 审计确认180独特训练图/284条目，无训练与验证、开发留出或旧域回归的精确图像重复；不能消除同视频近邻帧关联。

CC BY-NC4.0数据仅本地非商业研究；没有GitHub/界面工作，没有发送邮件或上传原图/权重。
'''
 (B/'第三十五轮双视角食材实验报告.md').write_text(text,encoding='utf-8')
 print(json.dumps({'selected':Path(s['weights']).name,'passed':d['development_and_regression_gates_passed'],'views':{v:r['counts'] for v,r in views.items()},'old8':old8},ensure_ascii=False))
