import json,csv
from pathlib import Path
b=Path(__file__).resolve().parent;r=b/'experiments/visor_adaptation_pilot'
c=json.loads((r/'comparison.json').read_text(encoding='utf-8'));a=json.loads((r/'data_audit.json').read_text(encoding='utf-8'));m=json.loads((b/'visor_adaptation_pilot/manifest.json').read_text(encoding='utf-8'))
with (r/'training/results.csv').open(encoding='utf-8-sig') as f:epochs=list(csv.DictReader(f))
lines=['# 第二十三轮识别实验：第一视角领域适配训练','','本轮直接训练案板识别，不改界面、不上传GitHub。起点是上一轮扩充YOLO11s，使用官方VISOR train中此前未用于13张诊断的参与者。','',
'## 事前数据与分组','','初始10人随机分组经纯元数据检查发现验证组没有完整案板标注，未启动模型。根据标注可用性在任何模型训练或预测前修订为：P10/P11训练，P15验证，P35测试。只有四位有效参与者，属于小规模适配试验，不应称普遍跨家庭可靠。', '',
'新样本70张训练、18张验证、3张测试。只接受案板class18全部exhaustive=y的标注，按1920x1080稀疏多边形转可见外接框；视频帧号间隔至少300，限制每视频/参与者数量，未按模型效果取样。此前13张失败诊断图及所有其参与者均未混入本次训练。', '',
'保留旧526张训练图，加新70张，独特训练图596张。新70张各曝光三次，因此训练列表736条、606个框；额外140条是显式重复训练曝光，不是新样本。旧178张背景图保留。验证/测试不曝光重复，未将缺案板标注当负图。', '',
'数据审计检查图像/标签、跨集合字节/像素重复及近似候选，新的VISOR按真实参与者分组；旧训练组名是未核实场景的占位，不能冒充全部源场景已独立审核。训练前只查看12张均匀抽取的训练标注缩略图，没有看保留测试图或预测。', '',
'## 冻结训练设置','','YOLO11s，30轮，640输入，batch8，AdamW lr0=0.0002、mosaic=0.1；从既有扩充权重微调。固定置信度0.5、匹配IoU0.5。验证筛选要求精确率>=0.75、召回>=0.60、召回增加>=0.20，全部通过才进行一次保留测试。测试使用同样条件。没有阈值/结构/训练轮数搜索。', '',
'## 结果','','|集合/模型|TP|FP|FN|标注匹配精确率|召回率|F1|','|---|---:|---:|---:|---:|---:|---:|']
for split,label in [('validation','新参与者验证'),('test','保留参与者测试'),('historical_regression','历史开发回归')]:
 if split not in c:continue
 for key,name in [('reference','适配前'),('candidate','适配后')]:
  x=c[split][key]['micro'];precision=f'{x["precision"]:.4f}' if x['tp']+x['fp'] else '—（无预测）'
  lines.append(f'|{label}/{name}|{x["tp"]}|{x["fp"]}|{x["fn"]}|{precision}|{x["recall"]:.4f}|{x["f1"]:.4f}|')
lines+=['',f'验证全部通过：{c["validation_all_passed"]}；逐条：{json.dumps(c["validation_checks"],ensure_ascii=False)}。',f'保留测试是否评估：{c["test_evaluated"]}。']
if c['test_evaluated']:lines.append(f'测试全部通过：{c["test_all_passed"]}；逐条：{json.dumps(c["test_checks"],ensure_ascii=False)}。')
else:lines.append('验证未过预定条件，未消费保留测试结果。')
lines+=['','## 限制与实际项目状态','','验证只有1位参与者18帧，测试只有1位3帧，帧彼此相关；即使数字很高也不等于真实后厨准确率。新增验证和测试只有案板正图，没有负场景误报率。仅案板定位，未由本轮证明食物分类、接触或用途违规判断。', '',
'原默认模型与注册表未改，不发布权重。素材为VISOR官方CC BY-NC4.0，仅本地非商业研究；图片/模型未上传Github。作者与来源见上一轮报告及 https://epic-kitchens.github.io/VISOR/site 。可见mask外接框与原完整物体框口径不同，预训练独立性未证明。', '',
f'训练完成{len(epochs)}轮；计划、样本指纹、比较和权重SHA保存在experiments/visor_adaptation_pilot。新测试如被本轮查看，未来再用不能称首次未见测试。']
(b/'第二十三轮识别实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'validation_passed':c['validation_all_passed'],'test_evaluated':c['test_evaluated'],'validation':{k:v['micro'] for k,v in c['validation'].items()},'test':{k:v['micro'] for k,v in c.get('test',{}).items()}},ensure_ascii=False))
