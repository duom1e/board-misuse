"""Conservative spatial rule engine. An overlap is evidence, not proof of contact."""
from dataclasses import dataclass
from typing import FrozenSet, Tuple

@dataclass(frozen=True)
class Detection:
    category: str
    confidence: float
    xyxy: Tuple[float,float,float,float]

@dataclass(frozen=True)
class BoardRule:
    board_id: str
    xyxy: Tuple[float,float,float,float]
    allowed_foods: FrozenSet[str]

def valid_box(box):
    return len(box)==4 and all(__import__('math').isfinite(x) for x in box) and box[2]>box[0] and box[3]>box[1]

def overlap_fraction(food,board):
    x1,y1,x2,y2=food;bx1,by1,bx2,by2=board
    intersection=max(0,min(x2,bx2)-max(x1,bx1))*max(0,min(y2,by2)-max(y1,by1))
    return intersection/((x2-x1)*(y2-y1))

def evaluate(boards,foods,min_confidence=.5,min_overlap=.8):
    if not 0<=min_confidence<=1 or not 0<=min_overlap<=1: raise ValueError('Invalid threshold')
    ids=[b.board_id for b in boards]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate board IDs')
    for b in boards:
        if not valid_box(b.xyxy) or not b.allowed_foods: raise ValueError('Invalid board rule')
    observed={b.board_id:[] for b in boards};ambiguous=set()
    for food in foods:
        if not valid_box(food.xyxy) or not 0<=food.confidence<=1: raise ValueError('Invalid detection')
        if food.confidence<min_confidence: continue
        x1,y1,x2,y2=food.xyxy;cx=(x1+x2)/2;cy=(y1+y2)/2
        candidates=[]
        for b in boards:
            bx1,by1,bx2,by2=b.xyxy
            fraction=overlap_fraction(food.xyxy,b.xyxy)
            if bx1<=cx<=bx2 and by1<=cy<=by2 and fraction>=min_overlap:
                candidates.append((b,fraction))
        if len(candidates)>1:
            ambiguous.update(b.board_id for b,_ in candidates);continue
        if len(candidates)==1:
            b,fraction=candidates[0]
            observed[b.board_id].append({'category':food.category,'confidence':food.confidence,
                'xyxy':list(food.xyxy),'food_box_overlap_fraction':fraction,
                'conflicts_with_rule':food.category not in b.allowed_foods})
    output=[]
    for b in boards:
        evidence=observed[b.board_id];conflict=any(e['conflicts_with_rule'] for e in evidence)
        status='suspected_misuse' if conflict else ('needs_review' if b.board_id in ambiguous or not evidence else 'no_supported_conflict')
        output.append({'board_id':b.board_id,'status':status,'allowed_foods':sorted(b.allowed_foods),
            'evidence':evidence,'ambiguous_food_assignment':b.board_id in ambiguous,
            'limitation':'2D rectangle overlap cannot prove food contact, cutting action, or complete compliance'})
    return output
