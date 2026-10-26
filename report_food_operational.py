import json
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'experiments/canteen_food_mixed_operational'
if __name__=='__main__':
 decision=json.loads((E/'decision.json').read_text(encoding='utf-8'))
 evaluations=json.loads((E/'checkpoint_validation.json').read_text(encoding='utf-8'))
 selected=decision['selected_for_development'];lines=[]
 for r in evaluations:
  c=r['validation']['counts'];o=r['operational'];p=c['precision']
  lines.append(f'|{Path(r["weights"]).name}|{c["tp"]}|{c["fp"]}|{c["fn"]}|{p if p is not None else "未定义"}|{o["classes"]["meat"]["recall"]:.3f}|{o["classes"]["vegetable"]["recall"]:.3f}|{o["macro_f1"]:.3f}|{o["feasible"]}|')
 oldbefore=decision['old_before'];oldafter=decision['old_after']
 rows=decision['old8_development']['rows'] if decision['old8_development'] else []
 if rows:
  gt={r['image']:r for r in json.loads((B/'canteen_food_boxes_v2/manifest.json').read_text(encoding='utf-8'))['records']}
  sheet=Image.new('RGB',(1668,738),'white')
  for i,r in enumerate(rows):
   im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
   for box in gt[r['image']]['boxes_xyxy']:draw.rectangle(box,outline='lime',width=2)
   for p in r['predictions']:
    draw.rectangle(p['xyxy'],outline='red',width=3);draw.text((p['xyxy'][0],max(0,p['xyxy'][1]-12)),f'{["meat","vegetable"][p["class"]]} {p["confidence"]:.2f}',fill='red')
   x=i%4*417;y=i//4*369;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+344),f'{r["seconds"]}s TP{r["tp"]} FP{r["fp"]} FN{r["fn"]}',fill='black')
  sheet.save(E/'selected_development.jpg')
 old8=str(decision['old8_development']['counts']) if rows else '预锁门槛未通过，未继续评估旧8张。'
 side=json.loads((E/'side_diagnostic.json').read_text(encoding='utf-8')) if (E/'side_diagnostic.json').exists() else None
 side_text=str(side['result']['counts']) if side else '未执行'
 text=f'''# 第三十四轮：按使用阈值选权重，并加入旧数据混训

2026-10-26。

## 实际改动

先检查本机Ultralytics 8.4.188的metrics.py：检测器默认fitness只使用mAP50-95，并不直接使用conf=.5时肉/菜的召回。因此“best”不等于本项目固定置信度下表现最好；这不是框架故障，也不意味着换选择标准就能解决所有漏检。

本轮在训练之前写入locked_plan.json：固定conf=.5和匹配IoU=.5，仅用既有7张开发验证图选权重。先要求总体精确率≥.75、肉召回≥2/3、菜召回≥.5、两张无明显块状食材图上误检框数0；再按肉/菜macroF1、最小类召回、精确率、较少误检排序。同分依固定文件名字典序先者。候选包括每10轮自动保存的权重及框架best/last，实际保存轮次以文件名中的0起算epoch为准。没有看旧8张的结果选模型。

从原始foodseg权重重新混训：128张固定随机抽取的旧训练图+37张后厨训练图（各重复曝光3次），共165张独特图/239训练条目。40轮，640像素，batch8，AdamW lr.0002，seed42。新增样本曝光不是新增独特图。锁定旧验证集随机64张作回归检查，mAP50允许下降最多.05。未使用旧测试集。与上一轮数据量、更新次数不同，不能把任何改善全归因选权重算法。

## 验证选权重结果

|权重|TP|FP|FN|精确率|肉召回|菜召回|macroF1|过操作门槛|
|---|---:|---:|---:|---|---:|---:|---:|---|
'''+ '\n'.join(lines)+f'''

选作开发候选：{Path(selected['weights']).name}。操作门槛与旧域回归合并通过：{decision['development_and_regression_gates_passed']}。

旧64张验证回归mAP50：{oldbefore['map50']:.4f}→{oldafter['map50']:.4f}；mAP50-95：{oldbefore['map50_95']:.4f}→{oldafter['map50_95']:.4f}。这些是跨置信度曲线指标，不能与表内固定阈值精确率混读。仅64图，不能证明旧能力全面保留。

旧8张开发诊断：{old8}

另8张既有侧面机位开发诊断（未用于本轮选权重）：{side_text}。同一次操作换视角，仍不等于独立厨房。侧面召回未改善，原机位4/6不能推广为换视角可用。

## 决策与边界

候选未采用、未注册，active_models.json哈希核对未改变：{decision['registry_unchanged']}。本轮只是改进训练与选择流程，不能称真实后厨错用识别已可用。7张验证图仅5处食材、两张无明显块状食材，门槛非常不稳定；训练近邻帧与验证来自同一操作，关联强。旧8张也已多次用于开发，只能算开发回归。没有预测不等于空板/合规。现有标注限定案板上的粗食材聚集区，案板外其他食材未完整标注，不是通用全食物真值。

下一步需独立操作/更多厨房数据验证，同时扩大肉/菜形态、遮挡和困难负样本覆盖。不能为追旧8图反复选阈值或宣称独立准确率。未执行大模型/VLM微调。

## 检查与资产

选择指标4项有针对性测试已通过：错类别误报归类、空板时段误报拒绝、肉召回门槛、零预测不能算成功。数据审计核对165张独特训练图、标签存在及取值，无训练/验证或旧域回归精确重复；近邻帧关联已明确，哈希检查不能排除它。

脚本run_food_mixed_selection.py、test_operational_selection.py、report_food_operational.py；实验目录experiments/canteen_food_mixed_operational包含预锁计划、数据清单、审计、每个候选的逐图验证预测、旧域前后指标、决策与权重。有旧8图诊断时另有selected_development.jpg（绿GT、红预测）。

CC BY-NC4.0真实后厨数据仅本机研究，未上传原图/权重、未发邮件。没有界面或GitHub工作。
'''
 (B/'第三十四轮混训与模型选择报告.md').write_text(text,encoding='utf-8')
 print(json.dumps({'selected':Path(selected['weights']).name,'gates_passed':decision['development_and_regression_gates_passed'],'old8':old8},ensure_ascii=False))
