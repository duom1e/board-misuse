"""Experimental coarse-box filter. Candidate locations only, never compliance.

`board_position_usable=False` means the board box is too coarse (e.g. covers most of the frame),
so "the food is inside the board box" is trivially true and carries no information. In that case
a would-be `board_candidate` is downgraded to `uncertain` with reason `board_box_too_coarse`
instead of pretending the geometry proved something.
"""
import math


def validate_box(box):
    if len(box) != 4 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in box):
        raise ValueError('Box must contain four finite coordinates')
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Box has no positive area')


def fraction_inside(food, region):
    validate_box(food)
    validate_box(region)
    intersection = max(0, min(food[2], region[2]) - max(food[0], region[0])) * \
                   max(0, min(food[3], region[3]) - max(food[1], region[1]))
    return intersection / ((food[2] - food[0]) * (food[3] - food[1]))


def locate(food, boards=(), containers=(), board_position_usable=True):
    validate_box(food)
    container_fraction = max((fraction_inside(food, c) for c in containers), default=0)
    board_fraction = max((fraction_inside(food, b) for b in boards), default=0)
    if container_fraction >= .8:
        location, reason = 'container_candidate', 'mostly_inside_container_box'
    elif container_fraction > .2:
        location, reason = 'uncertain', 'partial_container_overlap'
    elif board_fraction >= .8 and not board_position_usable:
        location, reason = 'uncertain', 'board_box_too_coarse'
    elif board_fraction >= .8:
        location, reason = 'board_candidate', 'mostly_inside_board_box'
    else:
        location, reason = 'uncertain', 'insufficient_board_box_coverage'
    return {'location': location, 'container_fraction': container_fraction, 'board_fraction': board_fraction,
            'location_reason': reason, 'board_position_usable': board_position_usable,
            'geometry_only': True, 'status': 'needs_review',
            'physical_contact_verified': False, 'compliance_determined': False}
