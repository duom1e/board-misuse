import json,csv
from pathlib import Path
b=Path(__file__).resolve().parent;r=b/'experiments/visor_adaptation_broad';data=b/'visor_adaptation_broad'
c=json.loads((r/'comparison.json').read_text(encoding='utf-8'));manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
with (r/'training/results.csv').open(encoding='utf-8-sig') as f:epochs=list(csv.DictReader(f))
sets={split:[row for row in manifest['new_records'] if row['split']==split] for split in ['train','val','test']}
lines=['# 第二十四轮识别实验：多参与者第一视角适配','','针对上一轮仅两位训练拍摄者的局限，本轮先查纯标注可用性，再固定参与者及候选，未按模型效果取样。使用相同既有扩充YOLO11s起点，20轮640输入、AdamW lr0.0002；不是上一轮的单变量消融。','',
'## 数据和隔离','','|用途|参与者数|VISOR独特图数|','|---|---:|---:|']
for split in ['train','val','test']:lines.append(f'|{split}|{len({x["participant"] for x in sets[split]})}|{len(sets[split])}|')
lines+=['',f'旧526张训练图保留，新VISOR训练图各曝光两次；独特训练图为{526+len(sets["train"])}，列表条目为{526+2*len(sets["train"])}，重复曝光不算新图。开发验证P03/P15，保留测试P01/P35。P01此前未训练或查看图片/预测；P35保留3帧此前未评估。旧13张诊断原图不入训练，但部分诊断参与者其他帧可补训练，所以旧诊断组不能再当独立测试。', '',
'原始稀疏标注class18，exhaustive=y，1920x1080多边形转可见外接框；元数据帧号间隔>=300，每人最多12个均匀选样（P15开发验证保留18）。下载与跨集重复/近似审计后冻结，未把缺标注当负图。旧训练源场景ID仍未验证。', '',
'## 固定阈值结果（confidence0.5，IoU0.5）','','|集合/模型|TP|FP|FN|标注匹配精确率|召回率|F1|','|---|---:|---:|---:|---:|---:|---:|']
for split,label in [('validation','开发验证'),('test','保留测试'),('historical_regression','旧开发回归')]:
 for key,entry in c.get(split,{}).items():
  m=entry['micro'];p=f'{m["precision"]:.4f}' if m['tp']+m['fp'] else '—（无预测）'
  lines.append(f'|{label}/{key}|{m["tp"]}|{m["fp"]}|{m["fn"]}|{p}|{m["recall"]:.4f}|{m["f1"]:.4f}|')
lines+=['',f'事前筛选要求P>=.75、R>=.60、R增加>=.20；验证全部通过：{c["validation_all_passed"]}。',f'保留测试是否评估：{c["test_evaluated"]}。']
if c['test_evaluated']:lines.append(f'测试同样条件是否全部通过：{c["test_all_passed"]}。')
else:lines.append('验证未通过，按计划没有消费保留测试结果。')
lines+=['','## 解释边界','','新VISOR验证/测试只取案板正图，没有负场景误报率。帧在参与者内相关；测试仅两位拍摄者，且P35只有3帧，不能据此宣称普遍后厨监控准确率。本轮只研究案板定位，未证明食品识别、接触及用途违规判断。', '',
'无论结果如何，未修改默认模型或阈值，不自动发布/注册候选。素材CC BY-NC4.0，仅本地非商业研究，未上传图像/权重或Github。作者与来源见 https://epic-kitchens.github.io/VISOR/site 。', '',
f'共训练{len(epochs)}轮。计划、权重SHA、样本指纹、比较结果保存在experiments/visor_adaptation_broad。']
(b/'第二十四轮识别实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'validation':{k:v['micro'] for k,v in c['validation'].items()},'test_evaluated':c['test_evaluated'],'test':{k:v['micro'] for k,v in c.get('test',{}).items()}},ensure_ascii=False))
