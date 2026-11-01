import json,time,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent
for _ in range(1200):
 p=B/'food_empty_label_clean_continue30_v1/status.json'
 if p.exists() and json.loads(p.read_text()).get('stage')=='completed':break
 time.sleep(2)
else:raise RuntimeError('Continuation not completed; no evaluation performed')
for script in ['report_clean_continue30.py','evaluate_clean_continue30_kitchen.py']:
 subprocess.run([sys.executable,str(B/script)],cwd=B,check=True)
for tag in ['training','continue30']:
 E=B/f'food_empty_label_clean_{tag}_v1';old=json.loads((E/'old_after.json').read_text());groups=json.loads((E/'group_results.json').read_text());probe=json.loads((E/'after.json').read_text());k=json.loads((B/f'clean_{tag}_kitchen_validation_v1/results.json').read_text())['rows'];food=[r for r in probe if r['reference']!='background'];bg=[r for r in probe if r['reference']=='background']
 text=f'# 清理空标签后的训练结果：{tag}\n\n623独特图片、631训练条目，隔离41张含目标或存疑空标签图片。training为15轮原foodseg初始化；continue30为其last追加30轮，优化器重启，累计45轮，不等于不中断45轮。训练960、评估640/.25，固定last。\n\n旧mAP50 {old["map50"]:.4f}；鸡肉开发匹配 {sum(r["tp"] for r in groups if r["split"]=="source_held_development")}/6；另一厨房匹配 {sum(r["class_match"] for r in k if r["model"]=="aligned")}/4；背景误帧 {sum(bool(r["predictions"]) for r in bg)}/23；食材检出 {sum(bool(r["predictions"]) for r in food)}/19。\n\n所有开发来源反复消费，不代表独立准确率。数据量和步数变化，不能将差异全归因空标签。默认模型不变，保留原数据和权重。\n'
 (E/'报告.md').write_text(text,encoding='utf-8')
p=B/'build_detector_training_ledger.py';s=p.read_text(encoding='utf-8');s=s.replace("'food_green_board_negative_training_v1']","'food_green_board_negative_training_v1','food_empty_label_clean_training_v1','food_empty_label_clean_continue30_v1']");p.write_text(s,encoding='utf-8');subprocess.run([sys.executable,str(p)],cwd=B,check=True)
for name in ['交接提示词.md','交接提示词-Codex.md','交接给dsh.md']:
 p=B/name;p.write_text('# 最新：清理后15+30轮已完成\n\n两个阶段已评估并更新报告和训练总表，见food_empty_label_clean_training_v1及food_empty_label_clean_continue30_v1/报告.md。累计45轮，优化器在续训时重启。默认未替换。下一步读两个报告并汇总，不重复训练。\n\n---\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
