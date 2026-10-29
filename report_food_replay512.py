"""Fixed-validation comparison after increasing original training replay."""
import json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent;E=B/'food_replay512_detector_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 plan=read(E/'locked_plan.json');prev=read(B/'food_balanced_detector_v1/locked_plan.json')
 assert plan['old_validation']==prev['old_validation']
 assert plan['negative_records']==prev['negative_records'] and plan['positive_records']==prev['positive_records']
 assert hashlib.sha256((B/'active_models.json').read_bytes()).hexdigest()==plan['registry_sha256']
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 train=(E/'train.txt').read_text().splitlines();val=(E/'val.txt').read_text().splitlines();assert not set(map(sha,train))&set(map(sha,val))
 lines=['# 扩大旧食材回放的真实检测训练','','同上一轮正负平衡配置：16负例、11已有检测框正例、15轮AdamW .0001、640/batch8、固定last、同64图验证；旧图从128增至512，总539训练图。每轮更新次数和样本比例也改变，因此不是等算力单变量因果实验。','', '|模型|训练背景误报/16|训练外背景误报/7|食材有候选/19|食材出现参考类/19|旧mAP50|旧mAP50-95|','|---|---:|---:|---:|---:|---:|---:|'];summary=[]
 for name,folder in [('原模型','food_balanced_detector_v1'),('128旧图平衡','food_balanced_detector_v1'),('512旧图平衡','food_replay512_detector_v1')]:
  d=B/folder;rows=read(d/('before.json' if name=='原模型' else 'after.json'));metrics=read(d/('old_before.json' if name=='原模型' else 'old_after.json'))
  bg=[r for r in rows if r['reference']=='background'];food=[r for r in rows if r['reference']!='background'];a=sum(bool(r['predictions']) for r in bg if r['trained_negative']);b=sum(bool(r['predictions']) for r in bg if not r['trained_negative']);c=sum(bool(r['predictions']) for r in food);t=sum(any(q['category']==r['reference'] for q in r['predictions']) for r in food)
  summary.append(dict(name=name,train_background_fp=a,held_background_fp=b,food_detected=c,expected_class_present=t,**metrics));lines.append(f'|{name}|{a}|{b}|{c}|{t}|{metrics["map50"]:.4f}|{metrics["map50_95"]:.4f}|')
 lines+=['','42裁剪为历史开发数据，正例无穷尽检测框，仅为类别存在性诊断，不是实例召回率。鸡肉及新增辣椒源未参加这几次训练，历史已消费不称盲测。训练/验证图片SHA无交集；默认注册表未改。','', '不自动采用，保留候选最后轮权重与全部逐图预测，后续按误报、食材漏报及旧域共同评估。训练已结束，没有VLM微调。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(E/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print('\n'.join(lines[4:9]))
if __name__=='__main__':main()
