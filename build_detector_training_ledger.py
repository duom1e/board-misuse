"""Audit actual detector training artifacts, not inference-only experiments."""
import json,csv,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
FOLDERS=['food_hard_negative_detector_v1','food_balanced_detector_v1','food_replay512_detector_v1','food_replay512_positive3_detector_v1','food_group_adaptation_v1','food_chicken_group_adaptation_v1','food_yellow_chicken_adaptation_v1','food_color_augmentation_v1','food_clsweight_detector_v1','food_bamboo_source_adaptation_v1','food_bamboo_replay1024_v1','food_frozen_backbone_v1','food_carrot_balance_v1','food_resolution960_v1','food_cleaning_negative960_v1','food_multiveg960_v1']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 rows=[]
 for folder in FOLDERS:
  e=B/folder;last=e/'training/weights/last.pt';metrics=e/'old_after.json';history=e/'training/results.csv'
  if not all(p.exists() for p in [last,metrics,history]):continue
  plan=read(e/'locked_plan.json');epochs=list(csv.DictReader(history.open(encoding='utf-8')));assert len(epochs)==15
  after=read(e/'after.json');food=[r for r in after if r['reference']!='background'];bg=[r for r in after if r['reference']=='background'];group=e/'group_results.json';held=None
  if group.exists():held=sum(r['tp'] for r in read(group) if r['split']=='source_held_development')
  rows.append(dict(experiment=folder,training_entries=plan['train_count'],completed_epochs=len(epochs),weight_sha256=hashlib.sha256(last.read_bytes()).hexdigest(),weight_bytes=last.stat().st_size,food_detected=sum(bool(r['predictions']) for r in food),expected_class=sum(any(q['category']==r['reference'] for q in r['predictions']) for r in food),background_false_candidates=sum(bool(r['predictions']) for r in bg),held_group_matches=held,**read(metrics)))
 lines=['# 检测器真实训练记录','','以下每行均核对了15轮results.csv、最后轮权重及SHA、训练计划和训练后回归结果。不是仅运行推理。尚未训练大模型。所有候选未自动替换默认。','','|实验|训练条目|完成轮数|旧mAP50|食材检出/19|参考类别/19|背景误帧/23|鸡肉组框/6|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f'|{r["experiment"]}|{r["training_entries"]}|{r["completed_epochs"]}|{r["map50"]:.4f}|{r["food_detected"]}|{r["expected_class"]}|{r["background_false_candidates"]}|{r["held_group_matches"] if r["held_group_matches"] is not None else "未运行"}|')
 lines+=['','重复曝光时训练条目不等于独特图片；食材裁剪检出不是IoU召回。鸡肉来源已反复消费作开发，不是独立测试。每轮旧验证集相同；跨实验训练设置和数据规模有变化，不将表格视为等算力因果比较。完整参数见各locked_plan和training/args.yaml，权重保存在本机。']
 (B/'检测器真实训练总表.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(B/'detector_training_ledger.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');print('Audited completed detector trainings:',len(rows))
if __name__=='__main__':main()
