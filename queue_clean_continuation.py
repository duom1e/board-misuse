import json,time,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent
status=B/'food_empty_label_clean_training_v1/status.json'
for _ in range(900):
 if status.exists() and json.loads(status.read_text()).get('stage')=='completed':break
 time.sleep(2)
else:raise RuntimeError('First stage not completed within30min; did not start continuation')
assert not (B/'food_empty_label_clean_continue30_v1').exists()
with (B/'food_empty_label_clean_continue30_v1.log').open('w',encoding='utf-8') as log:
 result=subprocess.run([sys.executable,str(B/'train_food_empty_label_continue30.py')],cwd=B,stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)
