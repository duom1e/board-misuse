"""Retain official per-image author, original URL and license metadata."""
import csv
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
RAW=BASE.parent/'work'/'datasets'

def main():
    manifest=json.loads((BASE/'board_dataset'/'manifest.json').read_text())
    selected={r['image_id'] for r in manifest['images']}
    found=[]
    for split in ('train','validation','test'):
        with (RAW/f'oi_meta_{split}.csv').open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                if r['ImageID'] in selected:
                    r['DatasetSplit']=split
                    found.append(r)
        print('metadata',split,len(found),flush=True)
    missing=selected-{r['ImageID'] for r in found}
    if missing:
        raise RuntimeError(f'Missing attribution: {missing}')
    with (BASE/'board_dataset'/'attribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(found[0]))
        writer.writeheader()
        writer.writerows(found)
    print('Attribution saved for',len(found),'images',flush=True)

if __name__=='__main__':
    main()
