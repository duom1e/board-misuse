"""Resolve model paths after placing local weights at the paths in model_manifest.json."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
path=root/'active_models.json'
registry=json.loads(path.read_text(encoding='utf-8-sig'))
for entry in registry.values():
    weight=Path(entry['weights'])
    if not weight.is_absolute(): entry['weights']=str(root/weight)
path.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
print('Registry configured. Missing weights:')
for key,entry in registry.items():
    if not Path(entry['weights']).is_file(): print(key,entry['weights'])
