"""Compare identical validation and crop cohorts under positive replay exposure."""
import json,hashlib
from pathlib import Path
from collections import Counter
B=Path(__file__).resolve().parent;E=B/'food_replay512_positive3_detector_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 p=read(E/'locked_plan.json');q=read(B/'food_replay512_detector_v1/locked_plan.json')
 assert p['old_training']==q['old_training'] and p['old_validation']==q['old_validation'] and p['positive_records']==q['positive_records'] and p['negative_records']==q['negative_records']
 entries=(E/'train.txt').read_text().splitlines();counts=Counter(entries)
 assert len(entries)==561 and len(counts)==539
 for r in p['positive_records']:assert counts[str(E/'images/train'/('positive_'+Path(r['image']).name))]==3
 def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
 assert not set(map(sha,entries))&set(map(sha,p['old_validation']))
 assert sha(B/'active_models.json')==p['registry_sha256']
 table=[]
 for name,folder,baseline in [('原模型','food_replay512_detector_v1',True),('正例1次','food_replay512_detector_v1',False),('正例3次','food_replay512_positive3_detector_v1',False)]:
  d=B/folder;rows=read(d/('before.json' if baseline else 'after.json'));m=read(d/('old_before.json' if baseline else 'old_after.json'));bg=[r for r in rows if r['reference']=='background'];food=[r for r in rows if r['reference']!='background']
  table.append(dict(name=name,train_bg_fp=sum(bool(r['predictions']) for r in bg if r['trained_negative']),held_bg_fp=sum(bool(r['predictions']) for r in bg if not r['trained_negative']),food_detected=sum(bool(r['predictions']) for r in food),expected_category=sum(any(x['category']==r['reference'] for x in r['predictions']) for r in food),**m))
 lines=['# 固定512旧图后的正例曝光训练对照','','沿用15轮AdamW .0001、640/batch8、seed42、固定last；512旧图、16背景、11有检测框正例均不变，只将11正例条目曝光由1次改3次。539独特图片，条目539→561；更新次数略增，不是等计算量单因素证明。64验证图片及其SHA完全相同，训练验证SHA无交，默认注册表未变。','','|模型|训练背景误报/16|外源背景误报/7|食材检出/19|出现参考类别/19|旧mAP50|旧mAP50-95|','|---|---:|---:|---:|---:|---:|---:|']
 for r in table:lines.append(f'|{r["name"]}|{r["train_bg_fp"]}|{r["held_bg_fp"]}|{r["food_detected"]}|{r["expected_category"]}|{r["map50"]:.4f}|{r["map50_95"]:.4f}|')
 lines+=['','裁剪只统计候选和类别出现，非完整检测IoU真值，不能称实例召回；来源历史已消费，非盲测。外源背景为本轮未训练鸡肉与新增辣椒视频。正例来自已消费肉末切配帧，不能转称独立测试。','', '所有权重保留本机，不自动替换默认。训练已完成，无大模型微调。后续应根据真实错例补不同来源的检测框，而非无期限调整这批样本曝光。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(E/'comparison.json').write_text(json.dumps(dict(table=table,unique_images=539,entries=561,adopted=False),ensure_ascii=False,indent=2),encoding='utf-8');print('\n'.join(lines[4:10]))
if __name__=='__main__':main()
