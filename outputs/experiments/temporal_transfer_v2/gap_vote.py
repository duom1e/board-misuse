from pathlib import Path
import json
from collections import deque
from score import iou
root=Path(__file__).resolve().parent
prior=root.parent/'temporal_counter_20261115'
sources={'白板':prior/'white_full','炒锅线':prior,'切鸡':root/'BV1264y1F7a6','食堂':root/'BV1bY4y1Y7v5'}
out=Path(r'D:/CodexProjects/board-misuse/outputs')
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
meta=next(s for s in load(out/'yellow_chicken_sources_v1/sources.json') if s['id']=='BV1Go4y1o7ym')
white={Path(r['image']).name:r['box'] for r in load(out/'experiments/white_round_holdout_20261115/annotations.json')}
anchors={'白板':{r['frame_index']:white[Path(r['image']).name] for r in meta['frames']},'切鸡':{r['frame']:r['reference'] for r in load(root/'score.json')['sources']['BV1264y1F7a6']['comparisons'][0]['anchors']}}
results=[]
for name,folder in sources.items():
    rows=load(folder/'positive_frames.json'); recent=deque(maxlen=5)
    metrics={k:{'events':0,'frames':0,'hits':[],'active':False} for k in (4,5)}
    for idx,row in enumerate(rows):
        boxes=[p['xyxy'] for p in row if p['confidence']>=.25]
        recent.append(boxes)
        votes=[(box,sum(any(iou(box,b)>=.5 for b in group) for group in recent)) for box in boxes] if len(recent)==5 else []
        for k,m in metrics.items():
            kept=[box for box,n in votes if n>=k]
            active=bool(kept)
            m['events']+=int(active and not m['active'])
            m['frames']+=int(active)
            m['active']=active
            gt=anchors.get(name,{}).get(idx)
            if gt and any(iou(box,gt)>=.5 for box in kept):m['hits'].append(idx)
    for k,m in metrics.items():
        results.append({'source':name,'required':k,'segments':m['events'],'frames':m['frames'],'hit_indices':m['hits']})
        print(name,k,m['events'],m['frames'],len(m['hits']))
(root/'gap_vote.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
