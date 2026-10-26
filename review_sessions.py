"""Local review drafts, bound to normalized image bytes; never training labels."""
import json
import math
import os
import uuid
from pathlib import Path
from annotation_io import sha

PURPOSES = {'肉类专用', '蔬菜专用', '允许肉类和蔬菜'}
MODELS = {'当前默认', 'LVIS实验候选'}


def validate_draft(value):
    if value.get('version') != 1: raise ValueError('不支持的复核进度版本。')
    boxes, rules = value.get('boxes'), value.get('board_rules')
    if not isinstance(boxes,list) or not isinstance(rules,list) or len(boxes)!=len(rules):
        raise ValueError('案板框与用途数量不一致。')
    for box,rule in zip(boxes,rules):
        if not isinstance(box,list) or len(box)!=4 or not all(isinstance(n,(int,float)) and not isinstance(n,bool) and math.isfinite(n) for n in box):
            raise ValueError('案板坐标不是有效数字。')
        if not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1): raise ValueError('案板框超出图片范围。')
        if rule is not None and rule not in PURPOSES: raise ValueError('未知案板用途。')
    if value.get('rule') not in PURPOSES | {''} or value.get('model') not in MODELS:
        raise ValueError('未知用途或模型选项。')
    if not isinstance(value.get('source_group'),str): raise ValueError('来源组格式错误。')
    return value


def save_draft(folder, boxes, rules, rule='', model='当前默认', source_group=''):
    folder=Path(folder)
    value={'version':1,'image_sha256':sha(folder/'image.png'),'source_sha256':sha(folder/'source.json'),
           'boxes':[list(b) for b in boxes],'board_rules':list(rules),'rule':rule,'model':model,'source_group':source_group,
           'status':'draft_only_not_annotation_or_violation_truth', 'requires_whole_image_review_on_restore':True}
    validate_draft(value)
    destination=folder/'review_draft.json'
    temp=folder/('draft_'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temp,destination)
    finally:
        if temp.exists(): temp.unlink()
    return destination


def load_draft(path):
    path=Path(path).resolve()
    value=validate_draft(json.loads(path.read_text(encoding='utf-8-sig')))
    # No path from the draft can redirect image loading to another local file.
    folder=path.parent
    if sha(folder/'image.png')!=value.get('image_sha256') or sha(folder/'source.json')!=value.get('source_sha256'):
        raise ValueError('图片或来源记录已变化，不能恢复旧坐标。')
    meta=json.loads((folder/'source.json').read_text(encoding='utf-8-sig'))
    if meta.get('normalized_sha256')!=value['image_sha256']: raise ValueError('工作图片与来源记录不一致。')
    return folder,value,meta
