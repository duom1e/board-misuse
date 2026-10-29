"""Same-probe comparison of negative-only and positive-balanced training."""
import json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
def read(e,n):return json.loads((e/n).read_text(encoding='utf-8'))
def main():
 folders=[B/'food_hard_negative_detector_v1',B/'food_balanced_detector_v1'];base=read(folders[0],'before.json');allrows=[base]+[read(e,'after.json') for e in folders];summaries=[]
 for name,rows in zip(['原foodseg','负例增强','正负平衡'],allrows):
  assert [r['id'] for r in rows]==[r['id'] for r in base]
  groups={}
  for k,rs in [('训练背景',[r for r in rows if r['trained_negative']]),('训练外来源背景',[r for r in rows if r['reference']=='background' and not r['trained_negative']]),('食材裁剪',[r for r in rows if r['reference']!='background'])]:
   groups[k]=dict(count=len(rs),any_detection=sum(bool(r['predictions']) for r in rs),expected_category=sum(any(p['category']==r['reference'] for p in r['predictions']) for r in rs))
  summaries.append(dict(name=name,groups=groups))
 old=[read(folders[0],'old_before.json')]+[read(e,'old_after.json') for e in folders]
 for e in folders:
  plan=read(e,'locked_plan.json');assert hashlib.sha256((B/'active_models.json').read_bytes()).hexdigest()==plan['registry_sha256']
 out=B/'食材检测两轮训练对照.md';lines=['# 食材检测器两轮真实训练对照','','2026-10-29。两次均从原foodseg开始，固定15轮、AdamW lr=.0001、640/batch8、seed42、固定last。第一轮128旧图+16背景裁剪=144；第二轮同配置另加11张已有检测框正例=155。均为YOLO全层训练，未微调大模型。64旧验证图完全相同，训练/验证图片SHA无重合。','','|模型|训练背景误报/16|训练外来源背景误报/7|食材有候选/19|食材出现参考类别/19|旧mAP50|','|---|---:|---:|---:|---:|---:|']
 for s,o in zip(summaries,old):
  g=s['groups'];lines.append(f'|{s["name"]}|{g["训练背景"]["any_detection"]}|{g["训练外来源背景"]["any_detection"]}|{g["食材裁剪"]["any_detection"]}|{g["食材裁剪"]["expected_category"]}|{o["map50"]:.4f}|')
 lines+=['','裁剪正例没有完整检测框真值，只统计有没有候选和参考类别出现，不能称实例召回率。训练外来源指鸡肉与新增辣椒视频，本项目此前均消费过，非盲测。第二轮新增正例来自肉末视频，不能把同源食材诊断当独立泛化。两轮样本数/更新次数略不同，不是严格等算力因果实验。','', '## 逐图类别变化','','原始before/after.json保留所有候选类别、置信度和框。默认模型未替换；是否采用须兼顾误报、食材漏报与旧域表现，不能只报告背景误报减少。所有真实训练均已结束，权重在各实验training/weights/last.pt。']
 out.write_text('\n'.join(lines)+'\n',encoding='utf-8');(B/'food_detector_training_comparison.json').write_text(json.dumps(dict(summaries=summaries,old_metrics=old,adopted=False),ensure_ascii=False,indent=2),encoding='utf-8');print('\n'.join(lines[4:11]))
if __name__=='__main__':main()
