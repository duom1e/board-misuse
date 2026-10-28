"""Strict localization-response validation; failures never mean an empty scene."""
import json
FIELDS=('boards','containers','food')
def validate_region_response(raw):
    errors=[]
    def error(code,path,message):errors.append({'code':code,'path':path,'message':message})
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate_key: '+key)
            result[key]=value
        return result
    text=raw.strip()
    if text.startswith('```'):
        lines=text.splitlines()
        if lines[0] in ('```json','```') and lines[-1]=='```':text='\n'.join(lines[1:-1])
    parsed=None
    try:parsed=json.loads(text,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('non_finite_json: '+x)))
    except (json.JSONDecodeError,ValueError) as exc:error('json_error','$',str(exc))
    valid_fields=[]
    if not errors:
        if not isinstance(parsed,dict):error('root_type','$','Expected an object')
        else:
            for field in sorted(set(parsed)-set(FIELDS)):error('unexpected_field','$.'+field,'Unexpected field')
            for field in FIELDS:
                path='$.'+field;before=len(errors)
                if field not in parsed:error('missing_field',path,'Required field is missing');continue
                boxes=parsed[field]
                if not isinstance(boxes,list):error('field_type',path,'Expected a list of boxes');continue
                if len(boxes)>6:error('box_count',path,'At most 6 boxes are allowed')
                for i,box in enumerate(boxes):
                    bp=f'{path}[{i}]'
                    if not isinstance(box,list) or len(box)!=4:
                        error('box_shape',bp,'Expected [x1,y1,x2,y2]; nested boxes are not expanded');continue
                    if any(type(v) is not int for v in box):
                        for j,v in enumerate(box):
                            if type(v) is not int:error('coordinate_type',f'{bp}[{j}]','Coordinate must be an integer, not bool, float or nested list')
                        continue
                    for j,v in enumerate(box):
                        if v<0 or v>1000:error('coordinate_range',f'{bp}[{j}]','Normalized coordinate must be between 0 and 1000')
                    if box[2]<=box[0] or box[3]<=box[1]:error('box_extent',bp,'Box must have positive width and height')
                if len(errors)==before:valid_fields.append(field)
    valid=not errors
    return {'schema_version':1,'valid':valid,'errors':errors,'diagnostic_valid_fields':valid_fields,
            'normalized_boxes':parsed if valid else None,
            'board_state':'unknown' if not valid else 'candidate_present' if parsed['boards'] else 'no_board_detected',
            'status':'needs_review','compliance_determined':False,'raw_response':raw}
