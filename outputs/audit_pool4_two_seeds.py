from pathlib import Path
import json
from collections import Counter
B=Path(__file__).resolve().parent
E=B/'experiments'
report=[]
for seed,stamp in [(42,'20261112'),(73,'seed73_20261115')]:
    for size in [640,960]:
        runs={}
        for arm in ['control','positive']:
            run=E/f'pool4_{arm}15_{stamp}'
            p=E/(run.name+'_probe')/'640.json' if size==640 else run/'predictions.json'
            rows=json.loads(p.read_text())
            assert len(rows)==84
            runs[arm]={Path(r['image']).name:(r['gt'] is not None,any(b['score']>=.5 and (r['gt'] is None or b['iou']>=.5) for b in r['boxes'])) for r in rows}
        assert runs['control'].keys()==runs['positive'].keys() and len(runs['control'])==84
        groups={}
        for arm,data in runs.items():
            groups[arm]={'hits':sum(pos and hit for pos,hit in data.values()),'false':sum(not pos and hit for pos,hit in data.values()),'hard_round':sum(hit for name,(pos,hit) in data.items() if pos and name.startswith(('BV1dHckzwEBA','BV1TS4y1K7ZH')))}
        changes=[{'frame':f,'source':f.split('_')[0],'positive_label':v[0],'control_detection':v[1],'candidate_detection':runs['positive'][f][1]} for f,v in runs['control'].items() if v!=runs['positive'][f]]
        report.append(dict(seed=seed,size=size,models=groups,changes=changes))
out=E/'pool4_positive15_seed73_20261115'
(out/'paired_audit.json').write_text(json.dumps(report,indent=2))
lines=['# 四来源补充训练的双种子对照','','固定同一84帧开发集，14张有板、70张无板。每组15轮、每轮34批；置信度与IoU门槛均为0.5。','','| 种子 | 输入 | 对照命中 | 补图命中 | 对照误报 | 补图误报 | 困难圆板命中（对照 / 补图） |','|---|---|---|---|---|---|---|']
for r in report:
    a,b=r['models'].values()
    lines.append(f"| {r['seed']} | {r['size']} | {a['hits']}/14 | {b['hits']}/14 | {a['false']}/70 | {b['false']}/70 | {a['hard_round']}/5、{b['hard_round']}/5 |")
lines.extend(['','逐帧变化及其视频来源见 paired_audit.json。两次训练使用同一批粗框与开发视频，不能据此估计新厨房的错用识别率。'])
(out/'报告.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines).encode('ascii','replace').decode())
