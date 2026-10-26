import json,csv
from pathlib import Path
b=Path(__file__).resolve().parent;r=b/'experiments/board_expanded_s_reviewed'
c=json.loads((r/'comparison.json').read_text(encoding='utf-8'));d=json.loads((r/'error_breakdown.json').read_text(encoding='utf-8'))
with (r/'training/results.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
times=[float(row['time']) for row in rows];elapsed=times[-1]+sum(times[i-1] for i in range(1,len(times)) if times[i]<times[i-1])
lines=['# 第二十一轮识别实验：新增案板标注数据','','本轮为识别数据和模型训练试验，不改界面、不上传GitHub。','',
'新增LVIS官方train许可记录为CC BY-SA 2.0的候选61张。与既有图像进行Flickr ID、字节哈希及dHash筛查；预览排除294297竹餐垫疑似错标，未改为负样本。实际新增60张、73个案板框，训练集466→526张、323→396框；原验证57图/28框与旧测试38图/19框保持不变。', '',
'新增框在640输入下短边<64有21个，64~128有5个，>=128有47个。四页预览已查看，不等同专家全分辨率标注审核。', '',
'沿用上一轮YOLO11s通用COCO初始权重、50轮、640、batch8、种子和优化器设置，与未扩充训练的YOLO11s比较。数据变多增加每轮更新次数，因此不是纯样本内容隔离试验。', '',
'## 开发验证结果（固定confidence=0.5、IoU=0.5）','','|模型|TP|FP|FN|精确率|召回率|F1|mAP50|','|---|---:|---:|---:|---:|---:|---:|---:|']
for key,name in [('reference','扩充前YOLO11s'),('candidate','扩充后YOLO11s')]:
    v=c['results'][key];m=v['fixed']['micro'];lines.append(f'|{name}|{m["tp"]}|{m["fp"]}|{m["fn"]}|{m["precision"]:.4f}|{m["recall"]:.4f}|{m["f1"]:.4f}|{v["map50"]:.4f}|')
lines+=['','预先固定筛选条件：P>=0.75、R>=0.50、R增加>=0.10、F1增加>=0.05，全部满足才通过开发筛选。',f'全部通过：{c["all_passed"]}。逐条结果：{json.dumps(c["criteria"],ensure_ascii=False)}。', '',
'## 按目标短边分组（探索性）','','|短边@640|标注数|扩充前检出|扩充后检出|','|---|---:|---:|---:|']
for group in ['below64','64to128','atleast128']:
    a=d['reference']['size_groups'][group];z=d['candidate']['size_groups'][group];lines.append(f'|{group}|{a["gt"]}|{a["matched"]}|{z["matched"]}|')
lines+=['','## 限制与保留记录','','本轮仅评估已多次开发使用的验证集，未对旧38测试图推理。即使开发筛选通过，也不代表新的独立测试或后厨监控效果；未自动替换默认模型、阈值或注册候选。', '',
'底图存在COCO预训练重叠可能；Flickr ID和哈希不证明独立场景；公开图片包含摆盘、备用板及家庭场景。训练图/权重不公开发布，保留来源与许可元数据。许可说明：https://creativecommons.org/licenses/by-sa/2.0/ 。作者身份未独立核实。', '',
f'训练完成{len(rows)}轮，CSV累计时间约{elapsed:.1f}秒；完整计划、数据审计、权重SHA、比较和漏检图在experiments/board_expanded_s_reviewed。', '',
'正式训练第20轮后CUDA unknown error中断，显卡新进程计算检查通过后从last.pt恢复，保留优化器与轮数；恢复可能改变数据加载随机序列，不能声称逐位复现。恢复详情recovery.json。启动时一次工作目录导入错误导致过早启动；已中止PID43156并保留中断目录，正式采用审图后独立目录重新训练，未利用中断运行选型或覆盖原实验。']
(b/'第二十一轮识别实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'all_passed':c['all_passed'],'reference':c['results']['reference']['fixed']['micro'],'candidate':c['results']['candidate']['fixed']['micro']},ensure_ascii=False))
