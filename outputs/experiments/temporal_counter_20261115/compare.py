from pathlib import Path
import json, math
from collections import deque
OUT=Path(__file__).resolve().parent
B=Path('D:/CodexProjects/board-misuse/outputs')
def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)
def segments(flags):
    events=[]
    begin=None
    for i,detected in enumerate(flags+[False]):
        if detected and begin is None: begin=i
        elif not detected and begin is not None:
            events.append([begin,i-1])
            begin=None
    return events
def evaluate(folder,arm,t,window,anchors):
    frames=load(folder/f'{arm}_frames.json')
    recent=deque(maxlen=window)
    flags=[]
    details=[]
    for i,frame in enumerate(frames):
        candidates=[box['xyxy'] for box in frame if box['confidence']>=t]
        recent.append(candidates)
        kept=[]
        if len(recent)==window:
            for box in candidates:
                if all(any(iou(box,prior)>=.5 for prior in previous) for previous in recent):
                    kept.append(box)
        flags.append(bool(kept))
        if i in anchors:
            details.append({'frame':i,'hit':any(iou(box,anchors[i])>=.5 for box in kept)})
    return {'frames':len(frames),'flagged':sum(flags),'events':segments(flags),'anchors':details}
sources=load(B/'yellow_chicken_sources_v1/sources.json')
meta=next(s for s in sources if s['id']=='BV1Go4y1o7ym')
labels={Path(c['image']).name:c for c in load(B/'experiments/white_round_holdout_20261115/annotations.json')}
anchors={}
for record in meta['frames']:
    label=labels[Path(record['image']).name]
    assert record['sha256']==label['sha']
    anchors[record['frame_index']]=label['box']
assert len(anchors)==12
data=[]
for arm in ['control','positive']:
    for t in [.5,.25,.1]:
        for window in [1,3,5,8,15]:
            neg=evaluate(OUT,arm,t,window,{})
            white=evaluate(OUT/'white_full',arm,t,window,anchors)
            data.append({'arm':arm,'threshold':t,'window':window,'stove':neg,'white':white})
            print(arm,t,window,len(neg['events']),neg['flagged'],sum(x['hit'] for x in white['anchors']),[x['frame'] for x in white['anchors'] if x['hit']])
(OUT/'gate_comparison.json').write_text(json.dumps(data,indent=2),encoding='utf-8')

