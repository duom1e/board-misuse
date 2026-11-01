"""Audit full-frame predictions centered inside manually reviewed human-only regions."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'human_region_regression_v1'
SPECS=[(12,'face',[590,35,830,415]),(24,'face',[240,42,397,280]),(30,'hand_interior',[380,40,520,140]),(31,'hand_interior',[480,45,630,155])]
def main():
 E.mkdir(exist_ok=False);cases=json.loads((B/'food_batch36_v2/locked_plan.json').read_text())['cases'];records=[]
 for i,kind,box in SPECS:
  r=cases[i];assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==r['sha256'];records.append(dict(index=i,kind=kind,region=box,**r,annotation='Assistant inspected original; approximate human-only region; NOT exhaustive background GT',training_eligible=False))
  im=Image.open(r['image']);ImageDraw.Draw(im).rectangle(box,outline='cyan',width=4);im.thumbnail((640,400));im.save(E/f'{i}_region.jpg')
 (E/'manifest.json').write_text(json.dumps(dict(records=records,scope='Selected after viewing model mistakes; diagnostic regression, not blind test. Center-in-region flags require visual interpretation; food elsewhere in frame remains valid.'),indent=2),encoding='utf-8')
 pred=json.loads((B/'fullframe_multiscale_comparison_v1/results.json').read_text());out=[]
 for model in pred:
  for r in records:
   row=next(q for q in model['rows'] if q['index']==r['index']);x1,y1,x2,y2=r['region'];flag=[]
   for q in row['predictions']:
    a,b,c,d=q['box'];cx=(a+c)/2;cy=(b+d)/2
    if x1<=cx<=x2 and y1<=cy<=y2:flag.append(q)
   out.append(dict(model=model['model'],index=r['index'],kind=r['kind'],flagged_predictions=flag))
 (E/'results.json').write_text(json.dumps(out,indent=2));print([(r['model'],r['index'],len(r['flagged_predictions'])) for r in out])
 lines=['# 人体区域误检固定回归','','两张人脸、两张手部内部区域，助手看原图后标注。选择发生在看过模型错误之后，属于错误回归集，不是独立验证；不用于训练。对完整画面推理，候选中心落在区域内则标记待复核，不删除检测结果，不做自动过滤。图中其他位置可以有合法食材。','','|模型|标记区域/4|标记框数|','|---|---:|---:|']
 for name in ['small','expanded']:
  rows=[r for r in out if r['model']==name];lines.append(f'|{name}|{sum(bool(r["flagged_predictions"]) for r in rows)}|{sum(len(r["flagged_predictions"]) for r in rows)}|')
 lines+=['','局部区域不穷尽人体、误检或背景。指标不可换算全画面准确率。此规则仅用于离线检查，不当作部署屏蔽规则；人体与食材会遮挡，不能简单删除所有相交食材框。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
