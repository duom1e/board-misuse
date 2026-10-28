import json
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'experiments/minced_training_fit_check'
if __name__=='__main__':
 d=json.loads((E/'result.json').read_text());m=json.loads((B/'minced_shape_annotation_v1/manifest.json').read_text(encoding='utf-8'));gt={r['image']:r for r in m['records']};rows=d['after']['rows'];sheet=Image.new('RGB',(1600,3*355),'white')
 for i,r in enumerate(rows):
  im=Image.open(r['image']).convert('RGB');dr=ImageDraw.Draw(im)
  for box in gt[r['image']]['boxes_xyxy']:dr.rectangle(box,outline='lime',width=3)
  for p in r['predictions']:
   dr.rectangle(p['xyxy'],outline='red',width=4);dr.text((p['xyxy'][0],max(0,p['xyxy'][1]-15)),f'{["meat","vegetable"][p["class"]]} {p["confidence"]:.2f}',fill='red')
  im.thumbnail((400,325));x=i%4*400;y=i//4*355;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+330),f'{r["seconds"]}s {r["class"]} TP{r["tp"]} FP{r["fp"]} FN{r["fn"]}',fill='black')
 sheet.save(E/'training_fit_review.jpg')
 a=d['before']['counts'];z=d['after']['counts'];classes=d['after_per_class']
 text=f'''# 新细碎形态小样本拟合校验

2026-10-27。此轮只回答“模型能否拟合已标注的11张图”，不测泛化、不替换默认。

## 事前方案

同一新来源的11图（5肉6菜）作为训练和训练后检查，train=val显式相同。原foodseg初始化，不混旧图，关闭几何、颜色与mosaic/mixup增强。固定120轮，每轮一个11图batch，nbs11不作64张名义批次累积；AdamW lr.001、lrf1、warmup0、freeze0、seed42。固定用最后一轮last.pt，不搜索检查点或阈值；conf=.5、IoU=.5。

训练拟合门槛预定P>=.9、R>=.9，肉/菜各R>=.8。只做数据读取与拟合能力诊断；相比前混训同时改变数据比例、增强、优化器步进和学习率，因此不能将结果归因单一变量。

## 训练拟合结果

|阶段|TP|FP|FN|精确率|召回|
|---|---:|---:|---:|---:|---:|
|原模型|{a['tp']}|{a['fp']}|{a['fn']}|{a['precision']}|{a['recall']:.4f}|
|固定最后一轮|{z['tp']}|{z['fp']}|{z['fn']}|{z['precision']}|{z['recall']:.4f}|

肉训练召回：{classes['meat']['recall']:.4f}；菜训练召回：{classes['vegetable']['recall']:.4f}。预设拟合门槛通过：{d['training_fit_gate_passed']}。

这些数字全是同11张训练图上的拟合，绝不是现场准确率、独立验证或错用识别成功。训练结束框架打印的val也使用同11图，不能将其中mAP当测试成绩。训练图可以被记忆；单助手粗框不是专家标签，能拟合也不认证标注正确。

## 如何解读

若通过，说明在此优化/采样条件下现有模型可以记住这些细碎肉菜样本；此前混训低拟合不应直接归结为模型完全不具备区分能力，但混训比例、增强、优化步进、学习率及验证选型的相对影响仍需受控比较。若未通过，应先审输入、标注和模型设置，不继续堆大混训。

本轮没有运行旧B站20图、保留剁椒来源、旧64图回归，也没有宣称旧能力保留。保留来源仍未消耗。纯11图模型可能遗忘其他食材，候选不注册、不替换默认；注册表SHA未改变：{d['registry_unchanged']}。没有降阈值、没有追加轮数或按结果重选模型。

目录experiments/minced_training_fit_check包含预锁计划、11图数据副本、训练、before/result.json及training_fit_review.jpg（绿GT/红预测）。脚本run_minced_overfit_check.py、report_minced_fit_check.py。素材/模型本机研究，不外发图片或权重，没有GitHub同步。
'''
 (B/'细碎形态小样本拟合校验报告.md').write_text(text,encoding='utf-8')
 print(json.dumps({'before':a,'after':z,'classes':classes,'training_fit_only':True,'fit_gate':d['training_fit_gate_passed']}))
