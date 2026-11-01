"""Re-evaluate both completed candidates on identical corn-inclusive development labels."""
import os,json,hashlib
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
from ultralytics import YOLO
B=Path(__file__).resolve().parent;E=B/'corn_label_baseline_comparison_v1'
def main():
 E.mkdir(exist_ok=False);rows=[]
 for name in ['food_clean_multiscale45_v1','food_fullmask_multiscale45_v1']:
  w=B/name/'training/weights/last.pt';m=YOLO(str(w))
  for group in ['old64','fishfruit30']:
   v=m.val(data=str(B/f'food_corn_validation_v1/{group}.yaml'),imgsz=640,rect=False,device=0,batch=8,workers=0,plots=False,project=str(E),name=name+'_'+group,verbose=False)
   rows.append(dict(model=name,group=group,map50=float(v.box.map50),map50_95=float(v.box.map),weight_sha256=hashlib.sha256(w.read_bytes()).hexdigest()))
 (E/'results.json').write_text(json.dumps(rows,indent=2));lines=['# 同一修正标签下重评','','验证图仍为历史64图与鱼水果30图；已补玉米及广义类别，不是新独立测试。所有模型以同标签640/rect=False评估，不能直接与旧标签指标解释为训练进步。','','|模型|组|mAP50|mAP50-95|','|---|---|---:|---:|']
 for r in rows:lines.append(f'|{r["model"]}|{r["group"]}|{r["map50"]:.4f}|{r["map50_95"]:.4f}|')
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(rows)
if __name__=='__main__':main()
