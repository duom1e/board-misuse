import json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent;E=B/'food_hard_negative_detector_v1'
def read(n):return json.loads((E/n).read_text(encoding='utf-8'))
def main():
 before=read('before.json');after=read('after.json');plan=read('locked_plan.json');ob=read('old_before.json');oa=read('old_after.json')
 assert len(before)==len(after)==42
 groups={}
 for name,pred in [('before',before),('after',after)]:
  for group,rs in [('trained_negative',[r for r in pred if r['trained_negative']]),('held_source_background',[r for r in pred if r['source'] in ['BV1264y1F7a6','BV1XN411q7XJ'] and r['reference']=='background']),('food_crops',[r for r in pred if r['reference']!='background'])]:
   groups.setdefault(group,{})[name]=dict(count=len(rs),any_detection=sum(bool(r['predictions']) for r in rs),expected_class_present=sum(any(p['category']==r['reference'] for p in r['predictions']) for r in rs))
 registry=hashlib.sha256((B/'active_models.json').read_bytes()).hexdigest();assert registry==plan['registry_sha256']
 result=dict(groups=groups,old_before=ob,old_after=oa,adopted=False,default_unchanged=True,scope='Crop presence/class diagnostic only; positive crop boxes not annotated, no localization accuracy claim.')
 (E/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# 手刀负例增强：YOLO检测器真实训练','','从原foodseg权重出发，全层训练15轮AdamW lr=.0001、640/batch8/seed42、固定last，未按验证表现挑权重。128张旧食材图+16张精确背景crop，共144训练图；64旧验证不训练。鸡肉和新增辣椒两个来源未参与本次训练，均为历史消费开发来源，非新盲测。','','|分组|样本数|原模型有候选|训练后有候选|','|---|---:|---:|---:|']
 for k,v in groups.items():lines.append(f'|{k}|{v["before"]["count"]}|{v["before"]["any_detection"]}|{v["after"]["any_detection"]}|')
 lines += ['',f'旧64图 mAP50：{ob["map50"]:.4f} → {oa["map50"]:.4f}；mAP50-95：{ob["map50_95"]:.4f} → {oa["map50_95"]:.4f}。','', '背景有候选表示误报；正例无候选表示本裁剪没有检出。正例只做裁剪类别/存在性诊断，不能称实例IoU准确率。所有裁剪预测同conf=.25、640、rect=False。旧mAP是另一个验证口径，不可和裁剪数混合。', '', '默认不自动采用：本次只有背景crop增量、规模很小，必须结合食材漏检和旧域退步评估。训练源中的正食材crop与背景来自相同视频，不能称独立源泛化；只有列出的两个来源在本次训练外。原视频不能因某crop为背景而整图标负。','', '真实训练权重在training/weights/last.pt；before/after.json逐图预测、locked_plan.json训练前配置、comparison.json分组对照。未训练大模型，未更新默认注册表。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
