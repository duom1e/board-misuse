"""Wait for completed training, evaluate fixed last, preserve failure status."""
import json,time,subprocess,sys,traceback
from pathlib import Path
B=Path(__file__).resolve().parent;E=B/'food_fullmask_multiscale45_v1';status=B/'fullmask_finalizer_status.json'
def save(stage,**kw):status.write_text(json.dumps(dict(stage=stage,**kw),ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 save('waiting_training')
 deadline=time.monotonic()+3*3600
 while time.monotonic()<deadline:
  p=E/'status.json'
  if p.exists() and json.loads(p.read_text()).get('stage')=='completed':break
  time.sleep(5)
 else:raise RuntimeError('Training not completed within3hours; did not evaluate incomplete weights')
 save('evaluating')
 for s in ['report_food_fullmask_multiscale45.py','evaluate_fullmask_multiscale45_kitchen.py','compare_fullmask_fit.py']:
  subprocess.run([sys.executable,str(B/s)],cwd=B,check=True)
 lines=['# 扩大官方训练数据的45轮多尺度结果','','4634独特图片/4642条目，通用YOLO11n起点，多尺度约640—1280，固定last。比前轮大幅增加数据和训练步数，不能当等算力比较。开发集反复消费，不是独立准确率。','','|模型|旧mAP50|鱼水果mAP50|鸡肉匹配/6|新厨房匹配/4|背景误帧/23|食材检出/19|','|---|---:|---:|---:|---:|---:|---:|']
 for folder,kitchen in [('food_clean_multiscale45_v1','clean_multiscale45_kitchen_validation_v1'),('food_fullmask_multiscale45_v1','fullmask_multiscale45_kitchen_validation_v1')]:
  d=B/folder;read=lambda n:json.loads((d/n).read_text(encoding='utf-8'));g=read('group_results.json');p=read('after.json');k=json.loads((B/kitchen/'results.json').read_text())['rows'];old=read('old_after.json')['map50'];aligned=read('aligned_after.json')['map50'];held=sum(r['tp'] for r in g if r['split']=='source_held_development');new=sum(r['class_match'] for r in k if r['model']=='aligned');bg=sum(bool(r['predictions']) for r in p if r['reference']=='background');food=sum(bool(r['predictions']) for r in p if r['reference']!='background');lines.append(f'|{folder}|{old:.4f}|{aligned:.4f}|{held}|{new}|{bg}|{food}|')
 lines+=['','默认模型不自动替换。训练组框640/960对比在fullmask_fit_comparison_v1/results.json，只作训练拟合诊断。自动报告只记录结果，需读数判断收益。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 p=B/'build_detector_training_ledger.py';s=p.read_text(encoding='utf-8');s=s.replace("'food_clean_multiscale45_v1']","'food_clean_multiscale45_v1','food_fullmask_multiscale45_v1']");p.write_text(s,encoding='utf-8');subprocess.run([sys.executable,str(p)],cwd=B,check=True)
 for n in ['交接提示词.md','交接提示词-Codex.md','交接给dsh.md']:
  p=B/n;p.write_text('# 最新：第28次扩大数据训练及评估完成\n\nfood_fullmask_multiscale45_v1/报告.md已生成与第27次的固定比较，fullmask_fit_comparison_v1为640/960训练组框对照，总表已更新。请先读结果判断，不重复训练；默认未切换，代码报告待上传GitHub。\n\n---\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
 save('completed')
if __name__=='__main__':
 try:main()
 except Exception:
  save('failed',error=traceback.format_exc());raise
