import json,sys
from pathlib import Path
base=Path(r'D:\CodexProjects\board-misuse\outputs')
m=json.loads((base/'github_upload_manifest.json').read_text(encoding='utf-8'))
files=m['files'][int(sys.argv[1]):int(sys.argv[1])+6]
print(json.dumps([{'path':r['path'],'mode':'100644','type':'blob','content':(Path(m['folder'])/r['path']).read_text(encoding='utf-8-sig')} for r in files],ensure_ascii=True))
