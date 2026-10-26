import json,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from PIL import Image,ImageDraw
base=Path(r'D:\CodexProjects\board-misuse\work\datasets\cn_commercial_kitchen');out=base/'preview_audit_20261026';out.mkdir(exist_ok=True)
files=json.loads((base/'remote_file_inventory_20261026.json').read_text(encoding='utf-8'))
selected=[r for r in files if r.get('type')=='file' and r['path'].endswith('.PNG')]
def fetch(r):
 path=out/r['path'];path.parent.mkdir(parents=True,exist_ok=True)
 try:
  if not path.exists():
   response=requests.get('https://hf-mirror.com/datasets/nova-dynamics/Chinese_Commercial_Kitchen_Manipulation_Dataset_Preview/resolve/main/'+r['path'],timeout=60);response.raise_for_status()
   if len(response.content)!=r['size']:raise ValueError('Size mismatch')
   tmp=path.with_suffix('.partial');tmp.write_bytes(response.content);tmp.replace(path)
  with Image.open(path) as im:im.verify()
  return {'path':r['path'],'local':str(path),'status':'downloaded'}
 except Exception as exc:return {'path':r['path'],'status':'failed','error':str(exc)}
results=[]
with ThreadPoolExecutor(max_workers=5) as pool:
 for future in as_completed([pool.submit(fetch,r) for r in selected]):
  results.append(future.result());print(len(results),'/',len(selected),flush=True)
(out/'manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
for task in ['task_01_cutting','task_02_stir_fry']:
 rows=sorted([r for r in results if r['status']=='downloaded' and r['path'].startswith(task)],key=lambda r:r['path'])
 sheet=Image.new('RGB',(1280,((len(rows)+3)//4)*220),'white')
 for i,r in enumerate(rows):
  im=Image.open(r['local']).convert('RGB');im.thumbnail((315,185));x=i%4*320;y=i//4*220;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x,y+190),r['path'].split('/')[-1],fill='black')
 sheet.save(out/(task+'_sheet.jpg'))
print('DONE',sum(r['status']=='downloaded' for r in results),flush=True)
