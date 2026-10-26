import requests,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
base=Path(r'D:\CodexProjects\board-misuse\work\datasets\cn_commercial_kitchen')
files=['task_01_cutting/egocentric_1080p.mp4','task_01_cutting/side_view.mp4']
def download(name):
 dest=base/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():
  r=requests.get('https://hf-mirror.com/datasets/nova-dynamics/Chinese_Commercial_Kitchen_Manipulation_Dataset_Preview/resolve/main/'+name,timeout=90,stream=True);r.raise_for_status()
  tmp=dest.with_suffix('.partial')
  with tmp.open('wb') as f:
   for chunk in r.iter_content(1024*1024):f.write(chunk)
  tmp.replace(dest)
 cap=cv2.VideoCapture(str(dest));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
 return {'file':str(dest),'fps':fps,'frames':count,'seconds':count/fps,'size':[width,height]}
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(download,files))
(base/'additional_cutting_videos.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(results,ensure_ascii=False))
