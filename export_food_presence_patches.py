"""Source-grouped local classification candidate export, with explicit limits."""
import json,hashlib,shutil,html
from pathlib import Path
from collections import Counter,defaultdict
B=Path(__file__).resolve().parent;E=B/'food_presence_patches_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 m=E/'reviewed_manifest.json';data=read(m);rows=data['records'];y=read(E/'yolo_results.json');v=read(E/'vlm_results.json')
 assert y['manifest_sha256']==v['manifest_sha256']==sha(m)
 ys={r['id']:r for r in y['rows']};vs={r['id']:r for r in v['rows']}
 assert len(ys)==len(vs)==len(rows)==20
 target=E/'classification_candidates';target.mkdir(exist_ok=False)
 counts=defaultdict(Counter);bg=[];food=[];cards=[];exports=[]
 for r in rows:
  assert sha(r['image'])==r['image_sha256'] and sha(r['crop'])==r['crop_sha256']
  assert r['training_scope']=='local_experimental_crop_classification_only' and not r['detection_training_eligible']
  folder=target/r['source_group']/r['label'];folder.mkdir(parents=True,exist_ok=True)
  dest=folder/Path(r['crop']).name;shutil.copyfile(r['crop'],dest);assert sha(dest)==r['crop_sha256']
  exports.append(dict(id=r['id'],file=str(dest.relative_to(target)),source_group=r['source_group'],class_name=r['label'],sha256=r['crop_sha256']))
  counts[r['source_group']][r['label']]+=1
  yp=ys[r['id']]['predictions'];vp=vs[r['id']]['opinion'];isbg=r['label']=='background'
  entry=dict(id=r['id'],yolo_categories=[p['category'] for p in yp],vlm_opinion=vp)
  (bg if isbg else food).append(entry)
  title=html.escape(r['id']);cats=', '.join(entry['yolo_categories']) or '没有候选'
  cards.append(f'<article><h3>{title}</h3><img src="{Path(r["crop"]).name}"><p>看图参考：{r["label"]}<br>YOLO：{cats}<br>大模型：{vp}</p><p>{r["source_group"]} · {r["seconds"]:.2f}s</p></article>')
 stats=dict(source_classes=counts,background_count=len(bg),background_yolo_false_positive=sum(bool(r['yolo_categories']) for r in bg),background_vlm_unknown=sum(r['vlm_opinion']=='unknown' for r in bg),food_count=len(food),food_yolo_any_proposal=sum(bool(r['yolo_categories']) for r in food),food_vlm_reference_consistent=sum(vs[r['id']]['opinion']==r['label'] for r in rows if r['label']!='background'),patch_sha_unique=len({r['crop_sha256'] for r in rows}),source_images=len({r['image_sha256'] for r in rows}),training_performed=False,background_rows=bg,food_rows=food)
 (E/'audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
 (target/'manifest.json').write_text(json.dumps(dict(records=exports,split_assigned=False,scope='Crop classification candidates only. Source directories are not train/test assignments. Previously consumed videos, nonexpert references. Do not create object boxes from these category labels.'),ensure_ascii=False,indent=2),encoding='utf-8')
 (E/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>手刀与食材成对样本</title><style>body{font:16px sans-serif;background:#eef2f5;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}article{background:white;padding:15px}img{width:100%;height:210px;object-fit:contain}</style><h1>手、刀与食材：20 个裁剪开发样本</h1><p>2 个已消费视频、4 张原图；助手看图标注，未经专家复核。人工裁剪，不代表全图检测能力。unknown 表示没有明确食材证据，不是合规结论。未训练、未改变默认模型。</p><main>'+''.join(cards)+'</main>',encoding='utf-8')
 report='''# 手、刀与食材成对样本批次

从两个已有视频各抽查 12 帧，共 24 帧；选择 4 张原图，手工裁出 20 个区域并逐张核对：11 个手/刀/案板背景、6 个蔬菜、3 个肉类。精确保留视频 SHA、原图 SHA、时间、帧号、裁剪坐标及裁剪 SHA。不是新增独立视频，两个来源以前均用于开发。

## 实际检查结果

|范围|YOLO 原默认食材模型|冻结 4B 原分类提示词|
|---|---|---|
|11 个背景裁剪|1 个手部误报肉，其余无候选|11 个 unknown|
|9 个食材裁剪|3 个有候选，6 个无候选|3 meat、6 vegetable，与本批参考一致|

YOLO 使用 conf=.25、imgsz=640、rect=False、CPU；4B 使用固定原分类提示词、512 缩图、12 token，精确裁剪不加边距。YOLO“有候选”只描述裁剪级检出，不是框 IoU 准确率。

不能报告大模型 100% 识别准确率：只有 4 张人工选择的原图，多个相关裁剪，且由人工提供目标范围，未测试自动定位。这也不能推翻上一批大模型把手刀判肉的失败；本批精确裁剪和上一批检测候选框并不等价。background → unknown 也只是无明确食材证据的弃答，不是识别出了具体手或刀类别。

## 数据如何使用

classification_candidates 按来源视频/类别保存 20 张本地实验分类候选，manifest 保存每张 SHA；没有自动划分 train/test。助手视觉核对通过，专家复核状态未完成。可以作为小规模分类训练的候选补充，不能直接作为 YOLO 检测标签，不能把原图当无食材负图。

当前不足以独立训练并验收三分类器：肉类仅 3 张且全来自一个视频；另一来源只有蔬菜和背景。应先补不同来源的肉类、手遮食材和手刀背景，再整源划分，不能随机拆相邻帧。已有来源已消费，任何后续成绩只能作为开发回归。

本轮未训练、未替换默认模型；没有下载或上传媒体。页面已生成并静态核查，未做浏览器交互验收。

文件：reviewed_manifest.json 是人工参考与来源；yolo_results.json、vlm_results.json 是固定模型输出；audit.json 是核验统计；index.html 是逐图对照；classification_candidates 是来源分组的本地分类候选包。
'''
 (E/'报告.md').write_text(report,encoding='utf-8');print(json.dumps({k:v for k,v in stats.items() if not k.endswith('_rows')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
