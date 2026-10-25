"""Image-only prototype. All policy must be explicitly configured."""
import argparse
import json
import os
from pathlib import Path
from PIL import Image,ImageDraw
from misuse_rules import BoardRule,Detection,evaluate,valid_box

BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work'/'yolo-config')
os.environ.setdefault('YOLO_AUTOINSTALL','False')
os.environ.setdefault('YOLO_OFFLINE','True')
os.environ.setdefault('PIP_CACHE_DIR',str(BASE.parent/'work'/'pip-cache'))

def selected_weights(metrics_file,fallback):
    registry=BASE/'active_models.json'
    if metrics_file=='vegetable_metrics.json' and registry.exists():
        return Path(json.loads(registry.read_text(encoding='utf-8-sig'))['vegetable']['weights'])
    if metrics_file=='baseline_metrics.json' and registry.exists():
        entries=json.loads(registry.read_text(encoding='utf-8-sig'))
        if 'board' in entries:
            return Path(entries['board']['weights'])
    path=BASE/metrics_file
    return Path(json.loads(path.read_text(encoding='utf-8-sig'))['weights']) if path.exists() else BASE/fallback

def food_image_size():
    registry=BASE/'active_models.json'
    return int(json.loads(registry.read_text(encoding='utf-8-sig'))['vegetable']['imgsz']) if registry.exists() else 640

def board_model_spec(which='board'):
    registry=BASE/'active_models.json'
    entries=json.loads(registry.read_text(encoding='utf-8-sig')) if registry.exists() else {}
    if which=='board' and which not in entries:
        return {'key':'board','weights':selected_weights('baseline_metrics.json','training_runs/board_baseline/weights/best.pt'),
            'imgsz':640,'status':'historical fallback'}
    if which not in entries or entries[which].get('classes')!=['cutting_board']:
        raise ValueError(f'{which} is not a registered single-class cutting-board detector')
    return {'key':which,'weights':Path(entries[which]['weights']),'imgsz':int(entries[which].get('imgsz',640)),
        'status':entries[which].get('status'),'selection_record':entries[which].get('selection_record')}

ALIAS_TARGETS={'plant':['plant_food','vegetable'],'animal':['animal_food','meat'],
    'meat':['animal_food','meat'],'seafood':['animal_food','meat'],
    'vegetable':['plant_food','vegetable'],'other':['other_food','processed_meat'],
    'prepared':['other_food','processed_meat'],'processed_meat':['processed_meat','other_food']}

def food_model_spec(which='vegetable'):
    """Which food detector to use, and which policy words it can actually answer.

    Any registry entry with a 'classes' list can be selected by its key. Aliases such as
    'meat' or 'plant' resolve to whatever class that model actually has, and the result.json
    records the mapping so nobody mistakes an alias for a detection capability.
    """
    registry=BASE/'active_models.json'
    entries=json.loads(registry.read_text(encoding='utf-8-sig')) if registry.exists() else {}
    if which=='vegetable':
        return {'key':'vegetable','weights':selected_weights('vegetable_metrics.json','training_runs/vegetable_baseline/weights/best.pt'),
            'imgsz':food_image_size(),'policy_names':{'meat','vegetable','seafood','fruit'},
            'aliases':{},'classes':['vegetable'],
            'notes':['single vegetable class; meat and seafood are declarations only']}
    if which not in entries:
        raise RuntimeError(f'No {which} model registered in active_models.json')
    entry=entries[which]
    classes=list(entry.get('classes') or [])
    if not classes:
        raise RuntimeError(f'Registry entry {which} does not declare its classes')
    aliases={}
    for alias,targets in ALIAS_TARGETS.items():
        for target in targets:
            if target in classes:
                aliases[alias]=target
                break
    notes=list(entry.get('policy_alias_notes') or [])
    notes += [f'{alias} is mapped to {target}, which is the closest class this model has'
              for alias,target in sorted(aliases.items()) if alias not in classes]
    return {'key':which,'weights':Path(entry['weights']),'imgsz':int(entry.get('imgsz',640)),
        'policy_names':set(classes)|set(aliases),'aliases':aliases,'classes':classes,'notes':notes}


def validate_policy(config):
    """Check policy before loading weights or creating result files."""
    if not isinstance(config,dict): raise ValueError('Policy must be a JSON object')
    for key,default in [('food_confidence',.5),('board_confidence',.5),('min_food_overlap',.8)]:
        value=config.get(key,default)
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not 0<value<=1:
            raise ValueError(f'{key} must be greater than 0 and at most 1')
    if config.get('boards') and config.get('all_detected_boards_allowed_foods'):
        raise ValueError('Choose configured regions OR a policy for all detected boards')
    spec=food_model_spec(config.get('food_model','vegetable'))
    allowed_names=spec['policy_names']
    ids=[]
    for r in config.get('boards',[]):
        if not isinstance(r.get('allowed_foods'),list) or not r['allowed_foods'] or not set(r['allowed_foods'])<=allowed_names:
            raise ValueError('Invalid allowed_foods in board rule')
        ids.append(r.get('id'))
        box=r.get('xyxy_normalized',[])
        if len(box)!=4 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and 0<=v<=1 for v in box) or not valid_box(box):
            raise ValueError('Board ROI must be a nonempty normalized rectangle')
    if any(not isinstance(i,str) or not i.strip() for i in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Each board needs a unique nonempty string ID')
    allowed=config.get('all_detected_boards_allowed_foods')
    if allowed and (not isinstance(allowed,list) or not set(allowed)<=allowed_names):
        raise ValueError('Invalid declared food policy')
    if not config.get('boards') and not allowed:
        raise ValueError('Explicit board policy required')
    if not config.get('boards'):board_model_spec(config.get('board_model','board'))
    return spec


def inspect_image(source,config,destination,models=None):
    from ultralytics import YOLO
    spec=validate_policy(config)
    canon=lambda food: spec['aliases'].get(food,food)
    def load(weights):
        key=str(Path(weights).resolve())
        if not Path(key).is_file(): raise FileNotFoundError(f'Local weights missing: {key}')
        if models is None: return YOLO(key)
        if key not in models: models[key]=YOLO(key)
        return models[key]
    with Image.open(source) as image: image=image.convert('RGB')
    iw,ih=image.size
    board_records=[]
    board_weights=None
    board_imgsz=None
    board_spec=None
    if config.get('boards'):
        for r in config['boards']:
            box=r['xyxy_normalized']
            if len(box)!=4 or not all(0<=v<=1 for v in box): raise ValueError('Board ROI must be normalized to 0..1')
            board_records.append((BoardRule(r['id'],(box[0]*iw,box[1]*ih,box[2]*iw,box[3]*ih),frozenset(canon(f) for f in r['allowed_foods'])),'configured_region'))
    elif config.get('all_detected_boards_allowed_foods'):
        board_spec=board_model_spec(config.get('board_model','board'))
        board_weights=board_spec['weights'];board_imgsz=board_spec['imgsz']
        model=load(board_weights)
        if list(model.names.values())!=['cutting_board']:raise RuntimeError('Board weights have the wrong class definition')
        result=model.predict(image,device=0,imgsz=board_imgsz,conf=config.get('board_confidence',.5),verbose=False)[0]
        boxes=sorted(result.boxes.xyxy.cpu().tolist(),key=lambda b:(b[0],b[1]))
        for i,box in enumerate(boxes,1):
            board_records.append((BoardRule(f'candidate_{i}',tuple(box),frozenset(canon(f) for f in config['all_detected_boards_allowed_foods'])),'model_candidate'))
    else: raise ValueError('Explicit board policy required')
    food_weights=spec['weights']
    food_model=load(food_weights)
    model_names=list(food_model.names.values())
    if spec['key']=='vegetable' and model_names!=['vegetable']:
        raise RuntimeError('Expected the trained single-class vegetable model')
    if spec['key']!='vegetable' and set(model_names)!=set(spec['classes']):
        raise RuntimeError(f'Registry says {spec["classes"]} but the weights contain {model_names}')
    result=food_model.predict(image,device=0,imgsz=spec['imgsz'],conf=config.get('food_confidence',.5),verbose=False)[0]
    foods=[Detection(model_names[int(c)],float(score),tuple(box)) for box,score,c in
        zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]
    rules=[b for b,_ in board_records]
    checks=evaluate(rules,foods,min_confidence=config.get('food_confidence',.5),min_overlap=config.get('min_food_overlap',.8))
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    report={'source':str(Path(source).resolve()),'overall_status':('suspected_misuse' if any(r['status']=='suspected_misuse' for r in checks) else 'needs_review'),
        'demo_not_site_policy':config.get('demo_not_site_policy',True),
        'board_model_weights':str(board_weights) if board_weights else None,'board_model_imgsz':board_imgsz,
        'board_model':board_spec['key'] if board_spec else 'configured_region',
        'board_model_status':board_spec.get('status') if board_spec else 'explicit caller-supplied region',
        'thresholds':{key:config.get(key,default) for key,default in [('food_confidence',.5),('board_confidence',.5),('min_food_overlap',.8)]},
        'food_model_weights':str(food_weights),'food_model_imgsz':spec['imgsz'],'food_model':spec['key'],
        'supported_food_categories':model_names,'policy_alias_notes':spec['notes'],
        'unsupported':['meat-vs-seafood separation','raw/cooked state','cutting action','cleaning history'],
        'board_candidates':[{'id':b.board_id,'xyxy':list(b.xyxy),'basis':basis} for b,basis in board_records],
        'food_detections':[{'category':f.category,'confidence':f.confidence,'xyxy':list(f.xyxy)} for f in foods],
        'checks':checks,'policy':config,
        'note':'A suspected alert is based on 2D rectangle overlap. It does not prove contact or actual cutting.'}
    draw=ImageDraw.Draw(image)
    for b,_ in board_records:
        status=next(r['status'] for r in checks if r['board_id']==b.board_id)
        draw.rectangle(b.xyxy,outline='red' if status=='suspected_misuse' else 'blue',width=4)
        draw.text((b.xyxy[0]+4,b.xyxy[1]+4),f'{b.board_id}: {status}',fill='red')
    for food in foods:
        draw.rectangle(food.xyxy,outline='lime',width=3)
        draw.text((food.xyxy[0]+3,food.xyxy[1]+18),f'{food.category} {food.confidence:.2f}',fill='green')
    image.save(destination/'evidence.jpg')
    (destination/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image',help='Local image path')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config',help='Explicit board policy JSON')
    group.add_argument('--allowed',nargs='+',help='Declare policy for ALL detected boards in this image')
    parser.add_argument('--food-model',default='vegetable',
        help='vegetable = legacy single-class model; any other value = a key in active_models.json (e.g. food_type, foodseg)')
    parser.add_argument('--food-confidence',type=float,default=None,
        help='Override food confidence. Calibrated: foodseg 0.50 precision-first / 0.35 recall-first; food_taxonomy 0.50 / 0.25')
    parser.add_argument('--board-confidence',type=float,default=None,help='Override board candidate confidence')
    parser.add_argument('--board-model',default=None,help='Registered board detector; board is default, board_lvis is experimental opt-in')
    parser.add_argument('--output',default=str(BASE/'misuse_results'/'latest'))
    args=parser.parse_args()
    if not Path(args.image).is_file(): parser.error('Image does not exist')
    if args.food_confidence is not None and not 0<args.food_confidence<1: parser.error('--food-confidence must be between 0 and 1')
    config=json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if args.config else {'all_detected_boards_allowed_foods':args.allowed}
    config.setdefault('food_model',args.food_model)
    if args.board_model is not None:config['board_model']=args.board_model
    if args.food_confidence is not None: config['food_confidence']=args.food_confidence
    if args.board_confidence is not None: config['board_confidence']=args.board_confidence
    report=inspect_image(args.image,config,args.output)
    print(json.dumps({'overall_status':report['overall_status'],'output':str(Path(args.output).resolve()),'checks':report['checks']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
