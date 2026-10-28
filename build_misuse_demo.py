"""Offline evidence viewer with explicit hypothetical food-use rules."""
import argparse,hashlib,html,json,shutil
from pathlib import Path
from PIL import Image,ImageDraw

def decide(row,allowed):
    validation=row.get('output_validation',row.get('validation',{}))
    if not validation.get('valid',False):return 'needs_review','结构未知，不能判断',[]
    regions=row.get('food_regions',row.get('regions',[]));evidence=[]
    for i,r in enumerate(regions):
        if r.get('location')!='board_candidate':continue
        category=r.get('food_category','unknown')
        members={'meat','vegetable'} if category=='mixed' else {category}
        if category in ('unknown','other') or not r.get('category_inference_performed'):continue
        evidence.append(dict(region=i,category=category,conflict=(not members<=set(allowed)) if allowed else None,box=r['box'],location_reason=r.get('location_reason'),contact_verified=False))
    if not allowed:return 'needs_review','未配置现场用途：仅展示食材与位置意见，不判断错用',evidence
    if any(e['conflict'] for e in evidence):return 'suspected_misuse','与演示假设不符；现场用途未核实，不能认定错用',evidence
    return ('no_supported_conflict','已有类别意见未显示冲突，不能证明合规',evidence) if evidence else ('needs_review','没有可用的板上食材类别证据',evidence)

def main():
 p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--allowed',choices=['vegetable','meat','both','unconfigured'],default='unconfigured');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('输出目录必须不存在，防止覆写')
 data=json.loads(a.results.read_text(encoding='utf-8'));a.output.mkdir(parents=True);(a.output/'evidence').mkdir();allowed=[] if a.allowed=='unconfigured' else ['meat','vegetable'] if a.allowed=='both' else [a.allowed];records=[];cards=[]
 for i,row in enumerate(data['rows']):
    state,reason,evidence=decide(row,allowed);source=Path(row['image']);name=f'evidence/{i:03}.jpg';shutil.copy2(source,a.output/name)
    time=row.get('frame_seconds');label=f'{time:.3f}秒' if isinstance(time,(int,float)) else f'样本{i+1}（时间见来源记录）'
    record=dict(index=i,time_seconds=time,source_image=str(source),source_image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),status=state,reason=reason,evidence=evidence,compliance_determined=False,demo_not_site_policy=True,review_required=True);records.append(record)
    captions={'suspected_misuse':'与演示假设不符','needs_review':'证据不足','no_supported_conflict':'未见支持冲突的证据'}
    im=Image.open(source).convert('RGB');draw=ImageDraw.Draw(im);v=row.get('output_validation',row.get('validation',{}))
    if v.get('valid'):
        for x,y,z,k in v['normalized_boxes']['boards']:draw.rectangle([x*im.width/1000,y*im.height/1000,z*im.width/1000,k*im.height/1000],outline='cyan',width=3)
        for region in row.get('food_regions',row.get('regions',[])):draw.rectangle(region['box'],outline='orange',width=3)
    overlay=f'evidence/{i:03}_overlay.jpg';im.save(a.output/overlay)
    cards.append(f'<article data-status="{state}"><h2>{html.escape(label)} · {captions[state]}</h2><img src="{name}"><p>{html.escape(reason)}</p><p>类别意见：{html.escape(", ".join(e["category"] for e in evidence) or "无")}</p><details><summary>证据与原图来源</summary><pre>{html.escape(json.dumps(record,ensure_ascii=False,indent=2))}</pre></details></article>')
    cards[-1]=cards[-1].replace(f'<img src="{name}">',f'<p>青框：案板候选；橙框：食材候选。框不证明接触。</p><img src="{overlay}"><details><summary>查看原图</summary><img src="{name}"></details>')
 plan=dict(allowed_foods=allowed,policy_configured=bool(allowed),scope='假设每个案板候选都使用相同用途；无板ID跟踪或现场用途核实' if allowed else '未配置用途，只展示观察',demo_not_site_policy=True,source_results=str(a.results.resolve()),source_results_sha256=hashlib.sha256(a.results.read_bytes()).hexdigest(),training_performed=False,records=records)
 (a.output/'events.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>案板食材与演示规则对照</title><style>body{font:16px system-ui;background:#f1f4f7;margin:24px;color:#172334}header,article{background:white;padding:20px;margin:16px auto;max-width:1000px;border-radius:12px}img{max-width:100%;max-height:450px}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:10px;margin:4px}article[data-status=suspected_misuse]{border-left:6px solid #d54b28}</style><header><h1>案板食材与演示规则对照</h1><p>演示假设：所有案板候选允许 '''+html.escape(', '.join(allowed))+'''。当前未配置真实现场用途；这不是已核实的现场规定，不能据此认定视频人员用错案板。</p><p>二维框不证明接触、生熟或清洗；疑似冲突需人工复核。未检出及未知都不代表合规。没有跟踪，同一过程的多个采样不能算多个违规事件。</p><button onclick="filter('all')">全部</button><button onclick="filter('suspected_misuse')">与演示假设不符</button><button onclick="filter('needs_review')">证据不足</button></header>'''+''.join(cards)+'''<script>function filter(s){document.querySelectorAll('article').forEach(x=>x.hidden=s!=='all'&&x.dataset.status!==s)}</script></html>'''
 if not allowed:
    page=page.replace('案板食材与演示规则对照','视频案板与食材观察报告').replace('演示假设：所有案板候选允许 。','未配置案板用途：当前仅展示观察结果。').replace('>与演示假设不符</button>','>假设不符（未启用）</button>')
 (a.output/'index.html').write_text(page,encoding='utf-8');print({s:sum(r['status']==s for r in records) for s in ['suspected_misuse','needs_review','no_supported_conflict']})
if __name__=='__main__':main()
