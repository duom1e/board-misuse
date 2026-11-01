"""Audit actual parameter changes and training image/label pairing."""
import os,json,csv,hashlib
from pathlib import Path
os.environ['YOLO_OFFLINE']='True'
from ultralytics import YOLO
import torch
B=Path(__file__).resolve().parent;E=B/'training_integrity_audit_v1'
def main():
 E.mkdir(exist_ok=False);baseline=YOLO(str(B/'training_runs/foodseg_2cls/weights/best.pt')).model.float().state_dict();out=[]
 for folder in ['food_chopping_chicken_training_v1','food_chopping_exposure5_training_v1','food_green_board_negative_training_v1']:
  e=B/folder;model=YOLO(str(e/'training/weights/last.pt')).model.float();state=model.state_dict();params=dict(model.named_parameters());changes=[k for k in params if k in baseline and not torch.equal(state[k],baseline[k])];paths=(e/'train.txt').read_text().splitlines();missing=[];invalid=[];empty=0
  for p in set(paths):
   label=Path(p.replace('\\images\\','\\labels\\').replace('/images/','/labels/')).with_suffix('.txt')
   if not label.exists():missing.append(str(label));continue
   lines=label.read_text().splitlines();empty+=not lines
   for line in lines:
    x=list(map(float,line.split()))
    if len(x)!=5 or x[0] not in [0,1] or not all(0<=v<=1 for v in x[1:]) or x[3]<=0 or x[4]<=0:invalid.append(str(label))
  history=list(csv.DictReader((e/'training/results.csv').open(encoding='utf-8')))
  out.append(dict(experiment=folder,changed_parameter_tensors=len(changes),total_parameter_tensors=len(params),training_entries=len(paths),unique_paths=len(set(paths)),empty_labels=empty,missing_labels=missing,invalid_labels=invalid,csv_epochs=len(history),weight_sha256=hashlib.sha256((e/'training/weights/last.pt').read_bytes()).hexdigest()))
 (E/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');lines=['# 最近三轮训练完整性核查','','逐项读取最后权重与原foodseg比较、解析实际train.txt和对应标签，核查15轮CSV。参数改变不证明模型质量。','','|实验|变化参数张量/总数|条目/独特路径|空标签|缺标签|非法标签|完成轮数|','|---|---:|---:|---:|---:|---:|---:|']
 for r in out:lines.append(f'|{r["experiment"]}|{r["changed_parameter_tensors"]}/{r["total_parameter_tensors"]}|{r["training_entries"]}/{r["unique_paths"]}|{r["empty_labels"]}|{len(r["missing_labels"])}|{len(r["invalid_labels"])}|{r["csv_epochs"]}|')
 lines+=['','空标签含有意加入的肉/菜背景样本。此检查只核实文件配对与数值格式，不证明标注语义正确。未更改任何模型、数据或默认配置。恢复训练的随机状态未做逐步复现检验。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
if __name__=='__main__':main()
