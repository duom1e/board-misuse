"""Fixed public example under two invented policies; not a violation benchmark."""
import argparse
import csv
import hashlib
import json
import uuid
from pathlib import Path
from check_board_misuse import BASE,inspect_image

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default=str(BASE/'board_dataset'/'images'/'test'/'37d8f468504ad26c.jpg'))
    parser.add_argument('--food-model',default='foodseg',help='Explicit registry entry; defaults to the current two-class food detector')
    parser.add_argument('--board-model',default='board',help='Registry board model; board_lvis is experimental opt-in')
    parser.add_argument('--roi',nargs=4,type=float,metavar=('LEFT','TOP','RIGHT','BOTTOM'),help='Explicit normalized board rectangle, replacing automatic board detection for this illustration')
    parser.add_argument('--prefix',default=None,help='New result prefix. Existing demonstrations are never overwritten.')
    args=parser.parse_args()
    source=Path(args.image).resolve()
    if not source.is_file():parser.error('Example image does not exist')
    prefix=args.prefix if args.prefix is not None else f'current_{uuid.uuid4().hex[:8]}_'
    if not prefix or any(c in prefix for c in '/\\:'):parser.error('Prefix must be a nonempty filename prefix')
    root=BASE/'misuse_results'
    destinations={name:root/f'{prefix}demo_{name}_only' for name in ['meat','vegetable']}
    summary_path=root/f'{prefix}summary.json'
    if summary_path.exists() or any(p.exists() for p in destinations.values()):
        parser.error('Output already exists; use a new prefix')
    provenance=None
    if source==(BASE/'board_dataset'/'images'/'test'/source.name).resolve():
        with (BASE/'board_dataset'/'attribution.csv').open(encoding='utf-8-sig') as f:
            provenance=next((r for r in csv.DictReader(f) if r['ImageID']==source.stem),None)
    summary={'note':'Same fixed image under two invented policies; a software illustration, not a real violation or accuracy evaluation.',
        'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'food_model':args.food_model,'cases':[]}
    models={}
    for name,dest in destinations.items():
        config={'demo_not_site_policy':True,'food_model':args.food_model,'board_model':args.board_model,
            'description':'Invented policy on an existing image; software illustration only.',
            'all_detected_boards_allowed_foods':[name],
            'board_confidence':.5,'food_confidence':.5,'min_food_overlap':.8}
        if args.roi:
            config.pop('all_detected_boards_allowed_foods')
            config['boards']=[{'id':'configured_board','xyxy_normalized':args.roi,'allowed_foods':[name]}]
            config['description']='Invented policy with an explicitly supplied board region; software illustration only.'
        result=inspect_image(source,config,dest,models=models)
        (dest/'rules.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        if provenance:
            (dest/'attribution.json').write_text(json.dumps({**provenance,'Derivative':'evidence.jpg: original photo plus model boxes'},ensure_ascii=False,indent=2),encoding='utf-8')
        summary['cases'].append({'declared_policy':name,'status':result['overall_status'],
            'board_model_weights':result['board_model_weights'],'food_model_weights':result['food_model_weights'],
            'thresholds':result['thresholds'],'output':str(dest)})
        print(name,result['overall_status'],flush=True)
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Summary: {summary_path}',flush=True)

if __name__=='__main__':main()
