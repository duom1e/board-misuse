from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
out=Path(__file__).resolve().parent
rows=json.loads((out/'gate_comparison.json').read_text())
def count_hits(r):return sum(a['hit'] for a in r['white']['anchors'])
selected=[r for r in rows if r['arm']=='positive' and r['threshold']==.25]
canvas=Image.new('RGB',(1100,785),'#faf8f1')
draw=ImageDraw.Draw(canvas)
def put(x,y,text,size,color='#183735',bold=False):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh'+('bd' if bold else '')+'.ttc',size)
    draw.text((x,y),text,font=font,fill=color)
draw.rectangle((0,0,1100,162),fill='#163c39')
put(44,27,'后厨案板定位  /  连续视频实验     2026.11.15',25,'#aed4c7')
put(44,78,'连续几帧都看到案板，才保留检测框？',42,'white',True)
put(44,195,'补图模型  ·  640 输入  ·  置信度 0.25  ·  两段视频均为 30 帧/秒',25)
put(45,268,'要求连续帧数',23)
put(288,268,'炒锅线候选段数 ↓',23)
put(705,268,'白板标注帧命中 ↑',23)
for i,row in enumerate(selected):
    y=330+i*65
    n=len(row['stove']['events']); h=count_hits(row)
    put(49,y,str(row['window'])+' 帧',27,bold=row['window']==5)
    draw.rectangle((290,y+8,290+7*n,y+30),fill='#ce7755')
    put(305+7*n,y-2,str(n),25)
    draw.rectangle((705,y+8,705+22*h,y+30),fill='#358775')
    put(719+22*h,y-2,str(h)+'/12',25)
put(44,684,'炒锅线约 117 秒，逐段抽看峰值框。白色圆板按 12 张原有参考框计分。',22)
put(44,726,'同一位置要求框重合度 ≥ 0.5。本次结果用于开发复核，错用仍需人工判断。',21)
canvas.save(out/'连续视频结果.png')
lines=['# 案板定位的连续视频评估','',
'两版冻结的种子73模型分别处理炒锅线 BV1b1ixB5ERA 的全部3503帧，以及切鸡视频 BV1Go4y1o7ym 的5196个实际可解码帧。每段均为30fps。白板视频容器元数据5197帧，与连续解码相差1帧；12张历史参考图以原记录帧号配对，并核对两份标注清单的图像SHA。炒锅线每秒抽看一帧，另复核两组0.25门槛共68个事件各自最高分帧的最高分框。对象包括锅、灶台、配菜容器与排烟罩。远处少量物体身份不明，峰值框复核优先于给整帧作绝对无板断言。','',
'每个当前候选框必须在最近 N 帧（含当前）中，每帧找到一个置信度过门槛且与当前框IoU至少0.5的框。连续有保留框的帧合为一段。该规则只使用过去帧；覆盖多个同时出现的框时，段数按整画面合并。白板只在12张有参考框的原定帧计分，命中还须与参考框IoU至少0.5。正片夹有成品盘及镜头切换，其他帧的候选不能直接当成真阳性。','',
'|模型|置信度|连续帧数|炒锅线出现段数|炒锅线保留帧|白板命中 / 12|','|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"|{ {'positive':'补图','control':'对照'}[r['arm']]}|{r['threshold']}|{r['window']}|{len(r['stove']['events'])}|{r['stove']['flagged']}|{count_hits(r)}|")
lines+=['','补图组0.25门槛从单帧改为连续5帧（约0.17秒），炒锅线31段降至1段，白板7/12降至6/12。唯一保留下来的炒锅线事件覆盖21帧，约0.70秒。在这两段开发视频上探索筛选参数，不能外推每小时误报率。此处评的是板的位置和候选连续性，错用还需要食材状态、接触和现场规则。','',
'原始预测、每帧窗口筛选结果的事件及锚点命中详情保存在 JSON 文件。用 run.py 和 run_white.py 生成原始预测，再执行 compare.py 复算对照。']
(out/'报告.md').write_text('\n'.join(lines),encoding='utf-8')
