"""Finalize training with source-held probes and full-frame human regression."""
import json,time,subprocess,sys,traceback
from pathlib import Path
B=Path(__file__).resolve().parent;E=B/'food_human_negative10_v1';S=B/'human_negative_finalizer_status.json'
def status(stage,**kwargs):S.write_text(json.dumps(dict(stage=stage,**kwargs),indent=2),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 status('waiting_training');deadline=time.monotonic()+3*3600
 while time.monotonic()<deadline:
  p=E/'status.json'
  if p.exists() and read(p).get('stage')=='completed':break
  time.sleep(5)
 else:raise RuntimeError('Training incomplete; no final evaluation')
 status('evaluating')
 for script in ['report_food_human_negative10.py','evaluate_human_negative10_kitchen.py','compare_fullframe_human_negative.py']:
  subprocess.run([sys.executable,str(B/script)],cwd=B,check=True)
 regions=read(B/'human_region_regression_v1/manifest.json')['records'];pred=read(B/'fullframe_human_negative_comparison_v1/results.json');human=[]
 for model in pred:
  for r in regions:
   row=next(q for q in model['rows'] if q['index']==r['index']);a,b,c,d=r['region'];flag=[q for q in row['predictions'] if a<=(q['box'][0]+q['box'][2])/2<=c and b<=(q['box'][1]+q['box'][3])/2<=d];human.append(dict(model=model['model'],index=r['index'],flagged_predictions=flag))
 (E/'human_region_results.json').write_text(json.dumps(human,indent=2))
 lines=['# 人体负样本继续训练10轮','','从第28次last继续10轮，优化器重启，lr=.0001，多尺度。原4642条目加6张其他来源已目视检查的负例各5次，4672条目/4640独特图片。新增图片与4个人体回归区域来源不同，但历史资料被消费，非独立盲测。固定last，640/.25/rect=False。','','|模型|旧mAP50|鸡肉/6|另一厨房/4|背景误帧/23|食材检出/19|人体标记区域/4|','|---|---:|---:|---:|---:|---:|---:|']
 for folder,kitchen,tag in [('food_fullmask_multiscale45_v1','fullmask_multiscale45_kitchen_validation_v1','small'),('food_human_negative10_v1','human_negative10_kitchen_validation_v1','expanded')]:
  e=B/folder;groups=read(e/'group_results.json');p=read(e/'after.json');k=read(B/kitchen/'results.json')['rows'];lines.append(f'|{folder}|{read(e/"old_after.json")["map50"]:.4f}|{sum(r["tp"] for r in groups if r["split"]=="source_held_development")}|{sum(r["class_match"] for r in k if r["model"]=="aligned")}|{sum(bool(r["predictions"]) for r in p if r["reference"]=="background")}|{sum(bool(r["predictions"]) for r in p if r["reference"]!="background")}|{sum(bool(r["flagged_predictions"]) for r in human if r["model"]==tag)}|')
 lines+=['','人体标记为完整画面候选中心落入已检查人体区域的诊断，不是全视频误检率。36帧左右叠图在fullframe_human_negative_comparison_v1/index.html，待进一步目视复核。默认模型不自动替换。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 p=B/'build_detector_training_ledger.py';s=p.read_text(encoding='utf-8');needle="'food_fullmask_corn_multiscale45_v1']";assert needle in s;s=s.replace(needle,"'food_fullmask_corn_multiscale45_v1','food_human_negative10_v1']");p.write_text(s,encoding='utf-8');subprocess.run([sys.executable,str(p)],cwd=B,check=True)
 for n in ['交接提示词.md','交接提示词-Codex.md','交接给dsh.md']:
  p=B/n;p.write_text('# 第30次训练与自动评估完成\n\nfood_human_negative10_v1/报告.md已比较食材、厨房、人体回归。fullframe_human_negative_comparison_v1已生成36帧对照，需目视复核。默认未变。不要重复训练，先读报告和人体结果；总表30次。\n\n---\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
 status('completed')
if __name__=='__main__':
 try:main()
 except Exception:status('failed',error=traceback.format_exc());raise
