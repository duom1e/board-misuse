import json
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'experiments/minced_replay12_trial'
if __name__=='__main__':
 d=json.loads((E/'decision.json').read_text());r=json.loads((E/'reserved_source_decision.json').read_text());fit=json.loads((E/'extra_training_fit.json').read_text());ref=json.loads((B/'experiments/minced_shape_adaptation_v1_reviewed/decision.json').read_text());oldfit=json.loads((B/'experiments/minced_shape_adaptation_v1_reviewed/extra_training_fit.json').read_text());lines=[]
 for name,x in [('曝光3次',ref),('曝光12次',d)]:
  q=x['bilibili20_summary'];lines.append(f'|{name}|{Path(x["selected"]["weights"]).name}|{q["tp"]}|{q["fp"]}|{q["fn"]}|{q["meat_predictions"]}|{x["old_after"]["map50"]:.4f}|{x["operational_gate"]}|')
 reserved_text='未通过释放门槛，保留来源没有运行任何模型预测。'
 if r['released']:
  reserved_text=''
  gt={x['image']:x for x in json.loads((B/'minced_shape_annotation_v1/manifest.json').read_text(encoding='utf-8'))['records']};sheet=Image.new('RGB',(1280,1925),'white')
  for j,name in enumerate(['previous','candidate']):
   out=r['results'][name];s=out['summary'];passed=s['recall']>=.5 and (s['precision'] or 0)>=.75 and s['meat_predictions']==0;reserved_text+=f'- {name}：TP{s["tp"]}/FP{s["fp"]}/FN{s["fn"]}，肉预测{s["meat_predictions"]}，预定菜类门槛={passed}。\n'
   for i,row in enumerate(out['result']['rows']):
    im=Image.open(row['image']).convert('RGB');dr=ImageDraw.Draw(im)
    for box in gt[row['image']]['boxes_xyxy']:dr.rectangle(box,outline='lime',width=3)
    for p in row['predictions']:
     dr.rectangle(p['xyxy'],outline='red',width=4);dr.text((p['xyxy'][0],max(0,p['xyxy'][1]-14)),f'{["meat","vegetable"][p["class"]]} {p["confidence"]:.2f}',fill='red')
    im.thumbnail((640,360));x=j*640;y=i*385;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+5,y+363),f'{name} {row["seconds"]}s TP{row["tp"]} FP{row["fp"]} FN{row["fn"]}',fill='black')
  sheet.save(E/'reserved_source_comparison.jpg')
 f=fit['result']['counts'];of=oldfit['result']['counts']
 text=f'''# 新细碎样本混训曝光比较

2026-10-27。

## 事前方案

同样591张不同训练图、初始foodseg、freeze0、AdamW lr.0001、640、batch8、seed42、30轮及原15图验证选型规则。只提高新11图重复曝光从3至12，其他源保持；条目由645增至744，每轮batch由81增至93，因此增加了更新次数，并非等计算量单因素因果证明。新11曝光占比约5.1%→17.7%，不是新增独特图片。

旧512图与64张回归验证、3份原标注清单SHA、关键训练设置均事前断言同参考，重复次数另有replay_audit。固定conf=.5、IoU=.5，不按B站分数换权重。旧域mAP50允许下降.05。

这次释放保留来源额外要求所选模型对新11训练图P/R>=.9、肉菜各R>=.8。该拟合门槛只用于决定是否释放保留来源，不用于重选权重；它是训练拟合，不是性能验证。

## 已暴露开发诊断

|方案|所选权重|B站TP|FP|FN|肉预测|旧64图mAP50|双机位操作门槛|
|---|---|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(lines)+f'''

新11训练图：原方案TP{of['tp']}/FP{of['fp']}/FN{of['fn']}，本轮TP{f['tp']}/FP{f['fp']}/FN{f['fn']}。拟合门槛={fit['fit_gate']}；本轮肉训练召回{fit['classes']['meat']['recall']:.4f}、菜训练召回{fit['classes']['vegetable']['recall']:.4f}。

操作门槛={d['operational_gate']}，旧域门槛={d['regression_gate']}，旧B站红椒诊断目标={d['bilibili20_gate']}。B站20图已暴露、多次用于开发，不是最终测试；少报肉必须同时正确识别菜才有意义。

## 保留来源

是否释放：{r['released']}。

{reserved_text}

预定菜类门槛P>=.75、R>=.5、肉框0；5图6个菜粗组框，只有一个视频且没有肉/空板，不足以证明整体系统或跨厨房可用。若释放，它从此成为已消费开发证据，不能以后再叫新鲜测试。44秒重遮挡图事前排除，未据结果挑图。

## 决策与文件

不采用、不注册、不作自动错用结论；默认注册表未改变：{d['registry_unchanged']}。没有旧域肉类全面保留或VLM、生熟、接触能力声明。新图是助手单评审粗区域；类别与动作/字幕仍有关联。

实验目录experiments/minced_replay12_trial保存计划、审计、所有候选、训练、decision、extra_training_fit和reserved_source_decision。原曝光3试验保留。原始图像、视频、权重均留本机，无GitHub同步或外发消息。
'''
 (B/'细碎样本混训曝光对照报告.md').write_text(text,encoding='utf-8')
 print(json.dumps({'selected':Path(d['selected']['weights']).name,'fit':f,'fit_gate':fit['fit_gate'],'old_map50':d['old_after']['map50'],'operational':d['operational_gate'],'bilibili':d['bilibili20_summary'],'reserved_released':r['released'],'reserved':{k:v['summary'] for k,v in r['results'].items()}}))
