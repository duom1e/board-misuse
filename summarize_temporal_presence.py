"""Ordered sampling diagnostic; never infer unsampled event durations."""
import json,csv
from pathlib import Path
B=Path(__file__).resolve().parent;E=B/'experiments/qwen4b_temporal_presence_v1'
def main():
    labels=json.loads((B/'temporal_presence_v1/manifest.json').read_text(encoding='utf-8'))['records']
    raw={r['case']:r for r in json.loads((E/'results.json').read_text(encoding='utf-8'))['rows']}
    heads=json.loads((B/'experiments/temporal_head_filter/results.json').read_text(encoding='utf-8'))['results']
    predictions={'4b':{k:'unknown' if 'pixel_boxes' not in r else 'present' if r['pixel_boxes']['boards'] else 'absent' for k,r in raw.items()}}
    predictions.update({n:{r['case']:r['predicted'] for r in v['presence_diagnostic']} for n,v in heads.items()})
    windows=list(dict.fromkeys(r['window'] for r in labels));timeline=[];summary={}
    for name,pred in predictions.items():
        summary[name]={}
        for window in windows:
            records=sorted([r for r in labels if r['window']==window],key=lambda r:r['seconds'])
            counts={k:0 for k in ['tp','fp','fn','tn','unknown','max_fn_sampling_streak','max_fp_sampling_streak']};fnrun=fprun=0
            for r in records:
                expected=r['expected_board'];actual=pred[r['case']]
                key='unknown' if actual=='unknown' else ('tp' if actual=='present' else 'fn') if expected=='present' else ('fp' if actual=='present' else 'tn')
                counts[key]+=1;fnrun=fnrun+1 if key=='fn' else 0;fprun=fprun+1 if key=='fp' else 0
                counts['max_fn_sampling_streak']=max(counts['max_fn_sampling_streak'],fnrun);counts['max_fp_sampling_streak']=max(counts['max_fp_sampling_streak'],fprun)
                timeline.append({'model':name,'window':window,'seconds':r['seconds'],'expected':expected,'predicted':actual,'outcome':key})
            summary[name][window]=counts
    (E/'temporal_summary.json').write_text(json.dumps({'summary':summary,'scope':'Uniform2s sampled known development windows; streak counts refer to observed samples only, no continuous duration or event-detection claim.'},indent=2),encoding='utf-8')
    with (E/'timeline.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(timeline[0]));writer.writeheader();writer.writerows(timeline)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
    for ax,window in zip(axes,windows):
        records=sorted([r for r in labels if r['window']==window],key=lambda r:r['seconds'])
        for y,name in enumerate(['label','4b','small','expanded']):
            for r in records:
                actual=r['expected_board'] if name=='label' else predictions[name][r['case']]
                error=name!='label' and actual!=r['expected_board']
                ax.scatter(r['seconds'],y,c={'present':'#159957','absent':'#aaa','unknown':'#ffaa00'}[actual],s=85,edgecolors='red' if error else 'white',linewidths=2)
        ax.set_yticks(range(4),['Assistant label','4B proposals','Small head filter','Expanded head filter']);ax.invert_yaxis();ax.set_xticks([r['seconds'] for r in records]);ax.set_title(window+' | green=present, gray=absent, red outline=mismatch');ax.set_xlabel('Video timestamp (seconds), sampled every 2 s');ax.grid(axis='x',alpha=.15)
    fig.savefig(E/'timeline.png',dpi=140);plt.close(fig);print(json.dumps(summary))
if __name__=='__main__':main()
