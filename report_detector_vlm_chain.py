import json
from pathlib import Path
from evaluate_food_group_boxes import iou
B=Path(__file__).resolve().parent;E=B/'detector_vlm_category_chain_v1'
def main():
 rows=json.loads((E/'results.json').read_text(encoding='utf-8'))['rows'];stats={}
 for r in rows:
  s=stats.setdefault(r['model'],dict(food_gt=0,geometry_matches=0,original_class_matches=0,review_class_matches=0,bg_images=0,bg_proposals=0,bg_known_food_opinions=0,bg_unknown_opinions=0))
  c=r['case'];ps=r['predictions']
  if c['kind']=='background':
   s['bg_images']+=1;s['bg_proposals']+=len(ps);s['bg_known_food_opinions']+=sum(q['review']['food_category'] in ['meat','vegetable','mixed','other'] for q in ps);s['bg_unknown_opinions']+=sum(q['review']['food_category']=='unknown' for q in ps);continue
  s['food_gt']+=len(c['boxes'])
  for key,field in [('geometry_matches',None),('original_class_matches','original'),('review_class_matches','review')]:
   used=set()
   for box in c['boxes']:
    candidates=[]
    for i,q in enumerate(ps):
     label=q['original_category'] if field=='original' else q['review']['food_category']
     score=iou(box,q['box'])
     if i not in used and score>=.5 and (field is None or label==c['category']):candidates.append((score,i))
    if candidates:used.add(max(candidates)[1]);s[key]+=1
 (E/'audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# 检测与大模型类别复核混合对照','','固定6张食材组ROI15框、23张背景裁剪，两种检测器均同conf=.25/640/rect=False。对每个实际候选原4B/原提示词/10%扩边复核，原标签保留；离线计算仅替换类别的假设结果，不改变几何或自动删框。','','|检测器|不看类别框匹配/15|原类别正确匹配|复核类别正确匹配|背景候选数|背景候选仍报食物|背景unknown|','|---|---:|---:|---:|---:|---:|---:|']
 for name,s in stats.items():lines.append(f'|{name}|{s["geometry_matches"]}|{s["original_class_matches"]}|{s["review_class_matches"]}|{s["bg_proposals"]}|{s["bg_known_food_opinions"]}|{s["bg_unknown_opinions"]}|')
 lines+=['','三种匹配各自采用类别约束下IoU>=.5一对一；复核不能修复几何没有覆盖的组框。背景数是预测候选数，不能与背景图片数混作同一分母。unknown只是弃答，不能说已正确识别非食物。','', '本批来源此前消费，候选模型训练过部分图/同源裁剪；6个ROI有手工范围，非完整自动后厨评估。此次定向开发结果不能称独立准确率、合规判定或已部署收益。原始候选、扩边范围、原文均保留，未训练或切换默认。','', '即使肉菜类别改善，仍须检查手/刀误报及真实食材漏检；不以这批结果自动覆盖生产类别。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
