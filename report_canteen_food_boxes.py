import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent
E=B/'experiments/canteen_food_boxes_v1'
data=json.loads((E/'comparison.json').read_text(encoding='utf-8'))
manifest=json.loads((B/'canteen_food_boxes_v1/manifest.json').read_text(encoding='utf-8'))
gt={r['image']:r for r in manifest['records']}
rows=data['after']['test']['rows']
sheet=Image.new('RGB',(834,len(rows)*369),'white')
for i,row in enumerate(rows):
 for j,key in enumerate(['before','after']):
  r=next(x for x in data[key]['test']['rows'] if x['image']==row['image'])
  im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
  for box in gt[r['image']]['boxes_xyxy']:draw.rectangle(box,outline='lime',width=2)
  for p in r['predictions']:
   draw.rectangle(p['xyxy'],outline='red',width=2)
   draw.text((p['xyxy'][0],max(0,p['xyxy'][1]-12)),f'{["meat","vegetable"][p["class"]]} {p["confidence"]:.2f}',fill='red')
  sheet.paste(im,(j*417,i*369));ImageDraw.Draw(sheet).text((j*417+4,i*369+343),f'{key} {row["seconds"]}s TP{r["tp"]} FP{r["fp"]} FN{r["fn"]}',fill='black')
sheet.save(E/'development_comparison.jpg')
lines=[]
for s in ['val','test']:
 for k in ['before','after']:
  c=data[k][s]['counts'];p='未定义' if c['precision'] is None else f'{c["precision"]:.3f}'
  lines.append(f'|{s}（开发）|{k}|{c["tp"]}|{c["fp"]}|{c["fn"]}|{p}|{c["recall"]:.3f}|')
text='''# 第三十二轮：真实后厨食材框标注与检测微调

日期：2026-10-26。

## 完成的工作

把上一轮36张局部图片补成可训练的YOLO框标注：29张食材阳性图、29个肉/蔬菜粗区域框、7张案板无明显块状食材图。所有标注叠图已逐图目视核对，修正93秒过松蔬菜框、65/70秒肉区域覆盖；刀面反光不标为肉。标注者为助手，未经食品专家复核。

标的是案板上同类食材聚集区域，不是逐片实例，也不是分割掩膜。框可能包含切片间隙及遮挡的刀和手。裁剪区域仍带部分台面食材，它们不在“案板上食材”标注范围内。因此只能作为固定ROI条件下的实验数据，不能直接当完整图片的全部食物真值或通用负样本。

## 训练与比较

从现有foodseg_2cls检测器微调完整检测模型，40轮、640像素、batch8、AdamW lr0=.0002、seed42。训练21图，验证7图。固定置信度.5、匹配IoU.5；不搜索阈值。不是大模型/VLM训练。自动选择验证最佳权重后，在旧8张已暴露时间留出图做开发诊断。没有把它们称为新鲜独立测试。

|集合|模型|TP|FP|FN|精确率|召回|
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'''

这些是检测框指标，不能与上一轮三类分类头的5/8准确率直接比较。没有预测只表示未检出，不表示空板、清洁、合规或没有错用。

## 限制与后续

单个视频、同一个机位，近邻帧及固定背景关联强。旧8张已用于发现问题，这次只能算开发回归。训练肉样本仍以整条肉为主，切片和碎块的训练覆盖没有因新增框而自动解决。当前没有混训旧菜品集，也没有检验跨域遗忘；不注册、不替换默认、不接入自动错用结论。

下一步优先补不同形态肉的训练标签及另一机位的诊断标签；同一操作的多视角仍不能作为独立厨房证据。随后在固定训练计划下混训，并保留更多独立操作/厨房作最终评估。

## 文件

- 数据与审计：canteen_food_boxes_v1/manifest.json、audit.json、data.yaml及三张标注复核图。
- 计划与逐图结果：experiments/canteen_food_boxes_v1/locked_plan.json、baseline.json、comparison.json。
- 可视化：experiments/canteen_food_boxes_v1/development_comparison.jpg，绿色标注框、红色预测框，左微调前、右微调后。
- 权重：experiments/canteen_food_boxes_v1/training/weights/best.pt，仅本机研究候选。
- 脚本：prepare_canteen_food_boxes.py、run_canteen_food_detector.py、report_canteen_food_boxes.py。

来源仍为Nova Dynamics公开后厨样本包，CC BY-NC4.0。未上传原图/权重，未发送邮件。
'''
(B/'第三十二轮食材检测实验报告.md').write_text(text,encoding='utf-8')
print(json.dumps({s:data['after'][s]['counts'] for s in ['val','test']}))
