"""Strict localization-response validation; failures never mean an empty scene.

A schema failure must stay diagnosable: every error carries a code, an exact JSON path and a
message, and the untouched raw response is kept (plus its sha256) so a later reviewer can see
what the model actually said. Distinct failure modes are kept apart on purpose:

  json_error          the whole payload is not valid JSON
  root_type           the payload is not an object
  unexpected_field    extra top-level field (whole response rejected)
  missing_field       one of boards/containers/food is absent
  field_type          a field is not a list
  box_count           more than MAX_BOXES boxes in one field
  box_nested          a box contains a nested list instead of four integers
  box_shape           a box is a list of the wrong length (and not nested)
  coordinate_type     a coordinate is not a plain int (bool/float/string)
  coordinate_range    a coordinate falls outside 0..1000
  box_extent_zero     zero width or zero height
  box_inverted        x2<x1 or y2<y1

None of these ever map to "no board", "empty board" or compliance.

Box quality is a separate, diagnostic-only judgement: a frame-filling box is a valid answer to
"is there a board" but useless for "is the food on the board". `box_quality()` reports that
without ever changing validity, and `degenerate_boxes` lists the offending indices.
"""
import hashlib
import json

FIELDS = ('boards', 'containers', 'food')
MAX_BOXES = 6
COORD_MIN, COORD_MAX = 0, 1000
DEGENERATE_AREA = 0.6  # 板面占画面 60% 以上视为框过粗（近景特写除外，一律提示人工框定）


def box_quality(boxes):
    """框可用性诊断：整幅/近整幅的框仍算"看见"，但位置不可用于空间规则。"""
    usable = []
    degenerate = []
    for index, box in enumerate(boxes):
        area = ((box[2] - box[0]) * (box[3] - box[1])) / float(COORD_MAX * COORD_MAX)
        entry = {'index': index, 'area_fraction': round(area, 3)}
        (degenerate if area >= DEGENERATE_AREA else usable).append(entry)
    return {
        'boxes': len(boxes),
        'usable_boxes': len(usable),
        'degenerate_boxes': degenerate,
        'position_usable': bool(usable),
        'reason': None if usable else ('no_boxes' if not boxes else 'all_boxes_cover_most_of_frame'),
    }


def validate_region_response(raw, allow_partial_fields=False):
    errors = []

    def error(code, path, message):
        errors.append({'code': code, 'path': path, 'message': message})

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate_key: ' + key)
            result[key] = value
        return result

    text = raw.strip().lstrip("\ufeff")
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[0] in ('```json', '```') and lines[-1] == '```':
            text = '\n'.join(lines[1:-1])

    parsed = None
    try:
        parsed = json.loads(
            text, object_pairs_hook=pairs,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError('non_finite_json: ' + x)))
    except (json.JSONDecodeError, ValueError) as exc:
        # 输出被 max_new_tokens 截断时报错信息千差万别，单独给一个可诊断的错误码
        truncated = text.lstrip().startswith('{') and not text.rstrip().endswith('}')
        if truncated:
            error('json_truncated', '$',
                  f'Response looks truncated (opens with "{{" but never closes, {len(text)} chars): {exc}')
        else:
            error('json_error', '$', str(exc))

    valid_fields = []
    field_counts = {}
    if not errors:
        if not isinstance(parsed, dict):
            error('root_type', '$', 'Expected an object')
        else:
            for field in sorted(set(parsed) - set(FIELDS)):
                error('unexpected_field', '$.' + field, 'Unexpected field')
            for field in FIELDS:
                path = '$.' + field
                before = len(errors)
                if field not in parsed:
                    error('missing_field', path, 'Required field is missing')
                    continue
                boxes = parsed[field]
                if not isinstance(boxes, list):
                    error('field_type', path, 'Expected a list of boxes')
                    continue
                field_counts[field] = len(boxes)
                if len(boxes) > MAX_BOXES:
                    error('box_count', path, f'At most {MAX_BOXES} boxes are allowed, got {len(boxes)}')
                for index, box in enumerate(boxes):
                    box_path = f'{path}[{index}]'
                    if not isinstance(box, list):
                        error('box_shape', box_path, 'Expected a list [x1,y1,x2,y2]')
                        continue
                    nested = [i for i, value in enumerate(box) if isinstance(value, (list, dict))]
                    if nested:
                        error('box_nested', box_path,
                              f'Nested coordinates at index {nested}; nested boxes are not expanded')
                        continue
                    if len(box) != 4:
                        error('box_shape', box_path, f'Expected [x1,y1,x2,y2], got {len(box)} values')
                        continue
                    bad_types = [i for i, value in enumerate(box) if type(value) is not int]
                    if bad_types:
                        for i in bad_types:
                            error('coordinate_type', f'{box_path}[{i}]',
                                  'Coordinate must be an integer, not bool, float or string')
                        continue
                    out_of_range = [i for i, value in enumerate(box) if value < COORD_MIN or value > COORD_MAX]
                    for i in out_of_range:
                        error('coordinate_range', f'{box_path}[{i}]',
                              f'Normalized coordinate must be between {COORD_MIN} and {COORD_MAX}')
                    if out_of_range:
                        continue
                    if box[2] == box[0] or box[3] == box[1]:
                        error('box_extent_zero', box_path, 'Box has zero width or height')
                    elif box[2] < box[0] or box[3] < box[1]:
                        error('box_inverted', box_path, 'Box is inverted: x2<x1 or y2<y1')
                if len(errors) == before:
                    valid_fields.append(field)

    valid = not errors
    by_code = {}
    for item in errors:
        by_code[item['code']] = by_code.get(item['code'], 0) + 1
    # 诊断（不影响有效性）：整幅或接近整幅的框，对"食材是否在板上"没有用处
    degenerate = []
    if valid:
        for field in FIELDS:
            for index, box in enumerate(parsed.get(field, [])):
                area = ((box[2] - box[0]) * (box[3] - box[1])) / float(COORD_MAX * COORD_MAX)
                if area >= DEGENERATE_AREA:
                    degenerate.append({'field': field, 'index': index, 'area_fraction': round(area, 3)})
    board_state = 'unknown' if not valid else 'candidate_present' if parsed['boards'] else 'no_board_detected'
    partial = False
    if allow_partial_fields and not valid:
        # 字段级隔离：只有 boards 字段可用时，仍可给出案板判断，但整体必须标记为部分有效
        boards_ok = 'boards' in valid_fields
        board_errors = [e for e in errors if e['path'].startswith('$.boards')]
        if boards_ok and not board_errors:
            board_state = 'candidate_present' if parsed['boards'] else 'no_board_detected'
            partial = True
    return {
        'schema_version': 3 if allow_partial_fields else 2,
        'valid': valid,
        'partial_fields': partial,
        'invalid_fields': [f for f in FIELDS if f not in valid_fields] if not valid else [],
        'errors': errors,
        'error_count': len(errors),
        'errors_by_code': by_code,
        'diagnostic_valid_fields': valid_fields,
        'field_counts': field_counts,
        'degenerate_boxes': degenerate,
        'normalized_boxes': parsed if valid else None,
        'board_state': board_state,
        'status': 'needs_review',
        'compliance_determined': False,
        'raw_response': raw,
        'raw_length': len(raw),
        'raw_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
    }
