"""Experimental coarse-box filter. Candidate locations only, never compliance."""
import math
def validate_box(box):
    if len(box)!=4 or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in box):
        raise ValueError('Box must contain four finite coordinates')
    if box[2]<=box[0] or box[3]<=box[1]: raise ValueError('Box has no positive area')
def fraction_inside(food,region):
    validate_box(food);validate_box(region)
    intersection=max(0,min(food[2],region[2])-max(food[0],region[0]))*max(0,min(food[3],region[3])-max(food[1],region[1]))
    return intersection/((food[2]-food[0])*(food[3]-food[1]))
def locate(food,boards=(),containers=()):
    validate_box(food)
    container_fraction=max((fraction_inside(food,c) for c in containers),default=0)
    board_fraction=max((fraction_inside(food,b) for b in boards),default=0)
    if container_fraction>=.8: location='container_candidate';reason='mostly_inside_container_box'
    elif container_fraction>.2: location='uncertain';reason='partial_container_overlap'
    elif board_fraction>=.8: location='board_candidate';reason='mostly_inside_board_box'
    else: location='uncertain';reason='insufficient_board_box_coverage'
    return {'location':location,'container_fraction':container_fraction,'board_fraction':board_fraction,
            'location_reason':reason,'geometry_only':True,
            'status':'needs_review','physical_contact_verified':False,'compliance_determined':False}
