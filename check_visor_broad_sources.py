import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from prepare_visor_adaptation_pilot import metadata_pick
b=Path(r'D:\CodexProjects\board-misuse\outputs');g=json.loads((b/'visor_train_source_listing.json').read_text(encoding='utf-8'))['groups']
people=['P02','P03','P04','P06','P07','P08','P10','P11','P12','P22','P24','P25','P26','P27','P28','P30','P32','P37'];rows=[]
with ThreadPoolExecutor(max_workers=6) as pool:
 futures=[pool.submit(metadata_pick,(p,g[p])) for p in people]
 for future in as_completed(futures):
  r=future.result();rows.append(r);print(r['participant'],len(r['records']),flush=True)
  (b/'visor_broad_train_feasibility.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('DONE',sum(bool(r['records']) for r in rows),'eligible participants',flush=True)
