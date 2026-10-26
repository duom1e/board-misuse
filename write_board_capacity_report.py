import json,csv
from pathlib import Path
base=Path(__file__).resolve().parent
run=base/'experiments/board_capacity_s_validation'
result=json.loads((run/'comparison.json').read_text(encoding='utf-8'))
errors=json.loads((run/'error_breakdown.json').read_text(encoding='utf-8'))
plan=json.loads((run/'locked_plan.json').read_text(encoding='utf-8'))
with (run/'training/results.csv').open(encoding='utf-8-sig') as f:epochs=list(csv.DictReader(f))
lines=['# 第二十轮识别实验：更大模型的训练对比','','本轮直接训练案板检测模型。固定50轮，640输入，batch8，AdamW；从本地COCO预训练yolo11s开始，和当前LVIS案板候选比较。不是只改界面，也未上传GitHub。','',
'只使用已有开发验证集57张、28个案板框作筛选；未对38张旧测试集运行预测。验证集已多次用于开发，不是独立现场测试。', '',
'## 固定置信度0.5、匹配IoU0.5结果','','|模型|TP|FP|FN|精确率|召回率|F1|mAP50|','|---|---:|---:|---:|---:|---:|---:|---:|']
for key,name in [('reference','原LVIS候选'),('candidate','新YOLO11s候选')]:
    row=result['results'][key];m=row['fixed']['micro']
    lines.append(f'|{name}|{m["tp"]}|{m["fp"]}|{m["fn"]}|{m["precision"]:.4f}|{m["recall"]:.4f}|{m["f1"]:.4f}|{row["map50"]:.4f}|')
lines+=['','## 预先写定的开发筛选条件','','P≥0.70，R≥0.50，召回提升≥0.10，F1提升≥0.05；全部同时满足。没有看结果后降低条件。', '',f'全部通过：{result["all_passed"]}。逐条结果：`{json.dumps(result["criteria"],ensure_ascii=False)}`。', '',
'## 按标注框短边分析（探索性，不是追加验收条件）','','|输入640下标注短边|标注数|原候选检出|新候选检出|','|---|---:|---:|---:|']
for bucket in ['below64','64to128','atleast128']:
    a=errors['reference']['size_groups'][bucket];b=errors['candidate']['size_groups'][bucket]
    lines.append(f'|{bucket}|{a["gt"]}|{a["matched"]}|{b["matched"]}|')
lines+=['','## 限制与结论','','这次比较的是可用候选方案：新模型从通用COCO权重开始，旧模型经过更早的案板微调，不能把全部差异归因于模型容量。数据仍主要来自公开图片，含非切菜中的备用案板、遮挡及暗光，现场任务还需区分板是否正在被使用。', '',
'不论开发筛选是否通过，本轮没有证明现场识别准确率，没有修改默认模型或阈值；需要另外来源且未用于开发的数据验证后才能考虑默认替换。没有启动额外参数搜索。', '',
f'固定训练完成{len(epochs)}轮；训练CSV末行累计时间约{float(epochs[-1]["time"]):.1f}秒。完整计划、权重SHA、数据审计、比较和漏检图在 experiments/board_capacity_s_validation。']
(base/'第二十轮识别实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'all_passed':result['all_passed'],'reference':result['results']['reference']['fixed']['micro'],'candidate':result['results']['candidate']['fixed']['micro']},ensure_ascii=False))
