"""Compare repaired full-evidence batch with historical summaries; development only."""
import json
from pathlib import Path
B=Path(__file__).resolve().parent
O=B/'experiments/combined_geometry_v2_all'
new=json.loads((O/'summary.json').read_text(encoding='utf-8'))
old={}
for n in ['combined_batch_v1','combined_batch_v3','combined_batch_v4','combined_batch_v5']:
    for row in json.loads((B/'experiments'/n/'summary.json').read_text(encoding='utf-8'))['rows']:
        old[(row['source'],row['frame'])]=row
changes=[]
counts={'frames':len(new['rows']),'label_yes':0,'hits':0,'label_no':0,'false_positive':0,'position_among_yes':0,'parse_failures':0}
for r in new['rows']:
    prev=old.get((r['source'],r['frame']))
    fields=['board_presence','board_source','position_usable','food_locations']
    if prev and any(prev.get(k)!=r.get(k) for k in fields):
        changes.append({'source':r['source'],'frame':r['frame'],'before':{k:prev.get(k) for k in fields},'after':{k:r.get(k) for k in fields}})
    counts['parse_failures']+=not r['vlm_valid']
    present=r['board_presence'].startswith('present')
    if r['label_usable'] and r['label_board']=='yes':
        counts['label_yes']+=1; counts['hits']+=present; counts['position_among_yes']+=r['position_usable']
    if r['label_usable'] and r['label_board']=='no':
        counts['label_no']+=1; counts['false_positive']+=present
(O/'comparison.json').write_text(json.dumps({'counts':counts,'changes':changes},ensure_ascii=False,indent=2),encoding='utf-8')
report='# 融合修复后五批重跑（2026-11-02）\n\n'+json.dumps(counts,ensure_ascii=False,indent=2)+'\n\n与历史记录相比 '+str(len(changes))+' 帧的存在性、来源、位置可用性或食物位置发生变化。详情 comparison.json，逐帧原始结果 results/。\n\n这些是已使用且部分看过模型输出后修正标签的开发帧，不是独立测试；位置可用仅代表粗框门槛通过，不是定位准确率或物理接触证明。此次重新运行模型，旧批量缺原始框，不能把所有差异单独归因于坐标修复。默认模型保持不变，没有新增训练。\n'
(O/'报告.md').write_text(report,encoding='utf-8')
print(json.dumps(counts,ensure_ascii=False));print('changed_frames',len(changes))
