import json,hashlib,html
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_presence_head_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 data=json.loads((E/'results.json').read_text(encoding='utf-8'));plan=json.loads((E/'locked_plan.json').read_text(encoding='utf-8'))
 m=B/'food_presence_patches_v2/reviewed_manifest.json';assert sha(m)==plan['manifest_sha256'];refs=json.loads(m.read_text(encoding='utf-8'))['records'];byid={r['id']:r for r in refs}
 results=[r for f in data['folds'] for r in f['rows']];assert len(results)==len({r['id'] for r in results})==36
 assert sha(B/'active_models.json')==plan['active_models_sha256']
 errors=[r for r in results if r['reference']!=r['prediction']];matrix={c:Counter(r['prediction'] for r in results if r['reference']==c) for c in plan['classes']}
 for r in refs:assert sha(r['crop'])==r['crop_sha256'] and sha(r['image'])==r['image_sha256']
 for f in plan['folds']:
  train=[byid[k] for k in f['train_ids']];held=[byid[k] for k in f['held_ids']]
  assert not ({r['source_video_sha256'] for r in train}&{r['source_video_sha256'] for r in held})
  assert not ({r['image_sha256'] for r in train}&{r['image_sha256'] for r in held})
  assert not ({r['crop_sha256'] for r in train}&{r['crop_sha256'] for r in held})
 sheet=Image.new('RGB',(1200,600),'white');d=ImageDraw.Draw(sheet)
 for i,r in enumerate(errors):
  im=Image.open(byid[r['id']]['crop']);im.thumbnail((290,220));x=i%4*300;y=i//4*300;sheet.paste(im,(x,y));d.text((x+3,y+225),r['id'],fill='black');d.text((x+3,y+245),r['reference']+' -> '+r['prediction'],fill='black')
 sheet.save(E/'errors.jpg')
 audit=dict(confusion=matrix,error_count=len(errors),errors=errors,source_frame_crop_overlap=False,all_source_hashes_verified=True,default_models_unchanged=True,decision='reject_automatic_adoption',reason='False food on knife, missed small meat, meat/vegetable cross-source errors; manual crops only and tiny consumed development data.')
 (E/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# 食材与背景分类层训练报告','','本轮新增 16 个经助手看图核对的裁剪，合并旧 20 个，共 4 个视频、8 张原图、36 个裁剪：背景 20、肉 10、蔬菜 6。来源、原图及裁剪 SHA 留存；均为以前消费过的视频，非专家参考。','', '## 训练与验证', '', '实际训练了 4 个三分类线性层：冻结原始 YOLO11s 特征，640 方形输入 rect=False，L2 标准化，Adam lr=.01，固定120轮、固定seed20261029、只保存最后一轮。每轮完整留出一个视频，未按相邻帧随机拆分，未根据结果调参。YOLO主干和大模型均未微调。','', '|留出视频|训练样本符合参考|留出样本符合参考|','|---|---:|---:|']
 for f in data['folds']:lines.append(f'|{f["held_source"]}|{f["train_matches"]}/{f["train_count"]}|{f["held_matches"]}/{f["held_count"]}|')
 lines+=['','所有训练样本均被拟合，但跨来源仍有 8 个错误。各折使用不同模型；不能把合计结果当作一个已部署模型的准确率。每折留出类别构成不同，两组留出源没有蔬菜，不能横向只比总分。','', '## 留出混淆计数（行是真值参考，列是预测）', '', '|参考|背景|肉|蔬菜|','|---|---:|---:|---:|']
 for c in plan['classes']:lines.append('|'+c+'|'+ '|'.join(str(matrix[c][k]) for k in plan['classes'])+'|')
 lines+=['','## 决策','','拒绝自动采用。20 个背景中 1 个刀面被误判为蔬菜；10 个肉中 2 个误判蔬菜、1 个误判背景；6 个蔬菜中 4 个误判肉。尤其是背景误报与小肉漏掉，不能靠此分类层直接删除候选。','', '这验证了跨来源识别仍是瓶颈：训练拟合好不等于新视频表现好。人工精确裁剪绕过了定位问题，36 张也不足以衡量真实后厨表现。来源视频已消费；预训练数据是否含这些画面未知。','', '下一步优先补不同来源的黄色鸡皮/白色脂肪、红绿辣椒和刀面，保留本批错误为固定开发回归。不要对这8个错例反复调参数后再称独立测试。现场用途规则与真实接触依旧未确认，本实验不输出案板错用结论。','', '## 交付文件','','- locked_plan.json：训练前固定的超参数、权重SHA和每折样本。','- *_head_last.pt：4个实验分类层权重，未接入默认流程。','- results.json：各折训练/留出结果、逐样本预测和损失。','- audit.json：原图/视频/裁剪跨折隔离核验及拒绝采用决策。','- errors.jpg：全部8个错误裁剪。','- index.html：36个样本的留出预测对照。','','没有训练在后台运行；active_models.json SHA 与训练前一致。页面未做浏览器交互验收。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 cards=[]
 for r in results:
  ref=byid[r['id']];cards.append(f'<article><h3>{html.escape(r["id"])}</h3><img src="{Path(ref["crop"]).as_uri()}"><p>参考：{r["reference"]}<br>留出预测：{r["prediction"]}<br>来源：{r["source_group"]}</p></article>')
 (E/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>食材背景分类训练留出结果</title><style>body{font:16px sans-serif;background:#edf2f5;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}article{background:white;padding:14px}img{width:100%;height:200px;object-fit:contain}</style><h1>4 个视频整源留出训练 · 36 个裁剪</h1><p>实验未采用。每个样本由未训练该来源视频的分类层预测；来源此前已消费，非盲测。人工裁剪与非专家参考，不是自动定位或错用识别准确率。</p><main>'+''.join(cards)+'</main>',encoding='utf-8')
 print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
