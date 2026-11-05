"""合体入口：一条命令同时跑"检测器线"和"VLM 线"，按先锁的规则融合成一份复核结果。

两条线各自擅长什么（2026-11-01 的实测结论）：
  · 检测器线（YOLO）：食物是什么（meat/vegetable）——它判断"有没有板"在真实后厨不可靠；
  · VLM 线（Qwen3-VL-4B 单问）：有没有板——49/49 可解析、2/2 命中，但框常常过粗；
融合规则（v2 修复坐标单位与回退，历史规则曾在开发集调整）：

  1. 板存在性：以 VLM 为准；VLM 说"没有"但检测器案板模型 conf≥0.5 检出，且显式开启 cross-check → board_uncertain。
  2. 板位置：优先 VLM 框且必须过 box_quality；过粗时退到检测器框；都不可用 → position_usable=false。
  3. 食物：只用检测器框（conf≥0.25），位置归属走 food_region_gate，并入 position_usable 护栏。
  4. 结果永远 needs_review；每条证据标注来自哪条线。

用法：
  python review_combined.py <图片> --output-dir <新目录> [--no-vlm]
批量（同进程加载一次模型）：
  python run_combined_batch.py --out NEW_OUTPUT_NAME
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from food_region_gate import locate, validate_box
from region_response_validation import box_quality, validate_region_response

BASE = Path(__file__).resolve().parent
BOARD_PROMPT = BASE / 'experiments' / 'split_prompts_v1' / 'prompt_boards.txt'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')


def load_detectors():
    import ultralytics
    from ultralytics import YOLO
    registry = json.loads((BASE / 'active_models.json').read_text(encoding='utf-8'))
    return {'board': YOLO(registry['board']['weights']), 'food': YOLO(registry['foodseg']['weights']),
            'registry': registry,
            'provenance': {'ultralytics_version': ultralytics.__version__,
                           'weights': {key: {'path': registry[key]['weights'],
                             'sha256': hashlib.sha256(Path(registry[key]['weights']).read_bytes()).hexdigest()}
                             for key in ('board', 'foodseg')}}}


def load_vlm():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    download = json.loads((BASE / 'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        download['folder'], dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(download['folder'], local_files_only=True)
    return {'model': model, 'processor': processor, 'revision': download.get('revision')}


def ask_vlm(image, vlm):
    import torch

    prompt = BOARD_PROMPT.read_text(encoding='utf-8')
    small = image.copy()
    small.thumbnail((512, 512))
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': small},
                                            {'type': 'text', 'text': prompt}]}]
    inputs = vlm['processor'].apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors='pt').to(vlm['model'].device)
    with torch.inference_mode():
        generated = vlm['model'].generate(**inputs, max_new_tokens=200, do_sample=False)
    return vlm['processor'].batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def parse_vlm_boards(answer):
    try:
        start, end = answer.find('{'), answer.rfind('}')
        if start < 0 or end < 0:
            return None, '响应里没有 JSON 对象'
        piece = json.loads(answer[start:end + 1])
        if not isinstance(piece.get('boards'), list):
            return None, '缺少 boards 列表'
        return piece['boards'], None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)[:200]


TILE_OVERLAP = 0.1


def split_tiles(image):
    """2×2 分块（带少量重叠），每块放大到 512。"""
    width, height = image.size
    result = []
    for row in (0, 1):
        for col in (0, 1):
            x1 = int(width * (col * 0.5 - (TILE_OVERLAP if col else 0)))
            y1 = int(height * (row * 0.5 - (TILE_OVERLAP if row else 0)))
            x2 = int(width * (col * 0.5 + 0.5 + (TILE_OVERLAP if not col else 0)))
            y2 = int(height * (row * 0.5 + 0.5 + (TILE_OVERLAP if not row else 0)))
            box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            piece = image.crop(box)
            if piece.width < 32 or piece.height < 32:
                continue
            result.append({'box': box, 'image': piece.resize((512, 512), Image.LANCZOS)})
    return result


def map_tile_boxes_to_frame(tile_box, boxes):
    """把分块内的 0..1000 归一化框映射回整帧像素坐标。"""
    x1, y1, x2, y2 = tile_box
    span_x, span_y = x2 - x1, y2 - y1
    mapped = []
    for box in boxes:
        mapped.append([x1 + box[0] / 1000 * span_x, y1 + box[1] / 1000 * span_y,
                       x1 + box[2] / 1000 * span_x, y1 + box[3] / 1000 * span_y])
    return mapped


def fuse(vlm_state, vlm_quality, vlm_boxes_px, yolo_boards, cross_check=False, tiling=None):
    """固定融合规则；返回 (board_presence, chosen_boxes, chosen_source)。

    cross_check 默认关闭：2026-11-01 在 49 帧开发集上实测，YOLO 案板交叉验证会把
    8 个"明确无板"的帧翻成 uncertain，误伤远大于收益（见 experiments/combined_batch_v1/）。

    tiling：整帧判"无板"时的分块补救结果（dict，含 boxes_px）。2026-11-04 实测：
    3 张深色圆板里整帧只出 2 张，分块补到 3/3，且 3 张无板图零误报。
    """
    if tiling and tiling.get('boxes_px'):
        return 'present_tiling_fallback', tiling['boxes_px'], 'vlm_tiling'
    if vlm_state == 'candidate_present':
        if vlm_quality and vlm_quality['position_usable']:
            return 'present', vlm_boxes_px, 'vlm'
        if yolo_boards:
            return 'present_vlm_box_coarse_using_yolo', yolo_boards, 'yolo_fallback'
        return 'present_position_coarse', vlm_boxes_px, 'vlm_box_coarse'
    if vlm_state == 'no_board_detected':
        if yolo_boards and cross_check:
            return 'uncertain_vlm_no_yolo_yes', yolo_boards, 'yolo_crosscheck'
        return 'absent', [], 'none'
    if vlm_state is None:  # --no-vlm
        return ('detector_only_present' if yolo_boards else 'detector_only_absent',
                yolo_boards, 'yolo' if yolo_boards else 'none')
    return 'unknown', (yolo_boards or []), ('yolo_fallback' if yolo_boards else 'none')


def position_evidence(boxes_px, width, height, source, presence):
    """Pixel boxes in; normalized quality check; only eligible boxes enter geometry."""
    boxes_px = boxes_px or []
    normalized = [[b[0]*1000/width, b[1]*1000/height,
                   b[2]*1000/width, b[3]*1000/height] for b in boxes_px]
    quality = box_quality(normalized)
    rejected = {entry['index'] for entry in quality['degenerate_boxes']}
    eligible = [b for i, b in enumerate(boxes_px) if i not in rejected]
    allowed = source in ('vlm', 'yolo', 'yolo_crosscheck', 'yolo_fallback', 'human', 'vlm_tiling')
    if presence == 'unknown':
        allowed = False
        quality['reason'] = 'board_presence_unknown'
    usable = bool(eligible) and allowed
    return quality, eligible if usable else [], usable


def validate_human_box(box):
    if not isinstance(box, (list, tuple)):
        raise ValueError('人工框必须是四个数值')
    validate_box(box)
    if any(isinstance(v, bool) or not 0 <= v <= 1000 for v in box):
        raise ValueError('人工框坐标必须在 0..1000')
    return box


def load_human_annotation(annotation_path, image_path):
    payload = json.loads(Path(annotation_path).read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict):
        raise ValueError('人工标注 JSON 必须是对象')
    if payload.get('normalized', True) is not True:
        raise ValueError('人工框必须使用 0..1000 归一化坐标')
    expected = payload.get('image_sha256')
    actual = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    if expected is not None and expected != actual:
        raise ValueError('人工标注 SHA 与当前图片不匹配，拒绝套用其他图片的框')
    box = payload.get('board', payload.get('board_box'))
    validate_human_box(box)
    return box


def analyze(image_path, detectors, vlm=None, no_vlm=False, food_confidence=0.25, board_confidence=0.5,
            draw_overlay=False, cross_check=False, human_box=None,
            tiling_fallback=False):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    # 人工框（归一化 0..1000）：有它就以它为准，不再让模型决定板在哪
    human_box_px = None
    if human_box is not None:
        validate_human_box(human_box)
        human_box_px = [human_box[0] * width / 1000, human_box[1] * height / 1000,
                        human_box[2] * width / 1000, human_box[3] * height / 1000]

    board_result = detectors['board'].predict(source=str(image_path), conf=board_confidence,
                                              imgsz=960, rect=True, device=0, verbose=False)[0]
    food_result = detectors['food'].predict(source=str(image_path), conf=food_confidence,
                                            imgsz=960, rect=True, device=0, verbose=False)[0]
    yolo_boards = [[round(float(v), 1) for v in box] for box in board_result.boxes.xyxy.cpu().tolist()] \
        if board_result.boxes is not None and len(board_result.boxes) else []
    food_model = detectors['food']
    yolo_food = [{'category': food_model.names[int(cls)], 'confidence': round(float(conf), 3),
                  'box': [round(float(v), 1) for v in box]}
                 for box, conf, cls in zip(food_result.boxes.xyxy.cpu().tolist(),
                                           food_result.boxes.conf.cpu().tolist(),
                                           food_result.boxes.cls.cpu().tolist())] \
        if food_result.boxes is not None and len(food_result.boxes) else []

    vlm_info = {'performed': False, 'board_state': None, 'boxes_normalized': None, 'raw': None,
                'valid': False, 'error': None}
    vlm_boxes_px, vlm_quality = None, None
    if not no_vlm and vlm is not None:
        answer = ask_vlm(image, vlm)
        boxes, error = parse_vlm_boards(answer)
        validation = validate_region_response(json.dumps({'boards': boxes, 'containers': [], 'food': []})) \
            if boxes is not None else None
        vlm_info = {'performed': True, 'revision': vlm.get('revision'), 'raw': answer,
                    'boxes_normalized': boxes, 'error': error,
                    'valid': bool(validation and validation['valid']),
                    'board_state': validation['board_state'] if validation else 'unknown',
                    'errors_by_code': validation['errors_by_code'] if validation else {}}
        if validation and validation['valid'] and boxes:
            vlm_quality = box_quality(boxes)
            vlm_boxes_px = [[b[0] * width / 1000, b[1] * height / 1000,
                             b[2] * width / 1000, b[3] * height / 1000] for b in boxes]

    tiling_used = None
    if (not human_box_px and tiling_fallback and vlm is not None
            and vlm_info['performed'] and vlm_info['board_state'] == 'no_board_detected'):
        # 整帧判"无板"时，用 2×2 分块补一次（2026-11-04 实测：圆板 3/3 找回，负例 0 误报）
        claimed = []
        tile_log = []
        for tile in split_tiles(image):
            answer = ask_vlm(tile['image'], vlm)
            boxes, _error = parse_vlm_boards(answer)
            if boxes:
                mapped = map_tile_boxes_to_frame(tile['box'], boxes)
                claimed.extend(mapped)
                tile_log.append({'tile_box': list(tile['box']), 'boxes_in_frame': mapped})
        tiling_used = {'tiles_checked': 4, 'tiles_with_board': len(tile_log),
                       'boxes_px': claimed, 'evidence': tile_log[:2]}

    if human_box_px:
        # 人工框优先：存在性、位置都以人工为准，模型只作为交叉参考
        board_presence, chosen_boxes, chosen_source = 'present_human_box', [human_box_px], 'human'
    else:
        board_presence, chosen_boxes, chosen_source = fuse(
            vlm_info['board_state'] if vlm_info['performed'] else None,
            vlm_quality, vlm_boxes_px, yolo_boards, cross_check=cross_check,
            tiling=tiling_used if (tiling_used and tiling_used['boxes_px']) else None)
    chosen_quality, geometry_boxes, board_position_usable = position_evidence(
        chosen_boxes, width, height, chosen_source, board_presence)

    food_regions = [{'category': item['category'], 'confidence': item['confidence'], 'box': item['box'],
                     **locate(item['box'], geometry_boxes or chosen_boxes, [], board_position_usable=board_position_usable)}
                    for item in yolo_food]

    result = {
        'provenance': {
            'detectors': detectors.get('provenance', {'recorded': False}),
            'inference': {'imgsz': 960, 'rect': True, 'device': 0,
                          'board_confidence': board_confidence, 'food_confidence': food_confidence,
                          'cross_check': cross_check, 'human_box_used': human_box is not None},
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'board_prompt_sha256': hashlib.sha256(BOARD_PROMPT.read_bytes()).hexdigest()
                if vlm_info['performed'] else None,
        },
        'status': 'needs_review',
        'compliance_determined': False,
        'image': str(Path(image_path).resolve()),
        'sha256': hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        'board_presence': board_presence,
        'board_absence_verified': False,
        'board_observation': ('not_detected' if board_presence in ('absent', 'detector_only_absent')
                              else 'candidate_or_uncertain'),
        'absence_note': '未检出不等于确认无板；检测器与大模型可能同时漏检。',
        'board_boxes': chosen_boxes,
        'board_geometry_boxes': geometry_boxes,
        'board_source': chosen_source,
        'position_usable': board_position_usable,
        'position_reason': chosen_quality.get('reason'),
        'food_from_detector': yolo_food,
        'food_regions': food_regions,
        'lines': {
            'detector': {'board_boxes': yolo_boards, 'board_conf_threshold': board_confidence,
                         'food_boxes': len(yolo_food), 'food_confidence': food_confidence},
            'vlm': vlm_info,
        },
        'fusion': {'version': '2_pixel_normalized_filtered', 'rules': 'vlm_presence_priority; vlm_box_if_tight; yolo_box_fallback; needs_review_always',
                   'vlm_box_quality': vlm_quality},
        'limitations': ('检测器在真实后厨判"有没有板"不可靠（conf 0.5 召回 0）；VLM 的框常覆盖整幅。'
                        '融合只解决"用谁的判断、用谁的框"，不产生准确率，也不判定合规。'),
    }
    image_out = None
    if draw_overlay:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(FONT), 18)
        for box in yolo_boards:
            draw.rectangle(box, outline='yellow', width=3)
            draw.text((box[0] + 3, box[1] + 3), 'yolo board', fill='yellow', font=font)
        for box in (vlm_boxes_px or []):
            draw.rectangle(box, outline='cyan', width=3)
            draw.text((box[0] + 3, box[1] + 24), 'vlm board', fill='cyan', font=font)
        for item in yolo_food:
            draw.rectangle(item['box'], outline='red', width=3)
            draw.text((item['box'][0] + 3, item['box'][1] + 3),
                      f"{item['category']} {item['confidence']:.2f}", fill='red', font=font)
        image_out = image
    return result, image_out


def report_text(result):
    lines = ['# 合体复核（检测器线 + VLM 线）', '',
             f"板存在性：**{result['board_presence']}**（来源：{result['board_source']}）",
             f"位置可用：{result['position_usable']}（{result['position_reason'] or 'ok'}）",
             '', '状态：needs_review，未判定合规。',
             '未检出案板不等于确认没有案板；两条线可能同时漏检。', '',
             f"检测器线：案板 {len(result['lines']['detector']['board_boxes'])} 框"
             f"（conf≥{result['lines']['detector']['board_conf_threshold']}）；"
             f"食物 {result['lines']['detector']['food_boxes']} 框",
             f"VLM 线：{result['lines']['vlm']['board_state'] if result['lines']['vlm']['performed'] else '未运行'}"
             + (f"（可解析={result['lines']['vlm']['valid']}）" if result['lines']['vlm']['performed'] else ''),
             '', '## 食物位置归属', '']
    if result['food_regions']:
        for region in result['food_regions']:
            lines.append(f"- {region['category']} {region['confidence']:.2f} → {region['location']}"
                         f"（{region['location_reason']}）")
    else:
        lines.append('- 检测器没有报出食物框；这不等于板上没有食物。')
    lines += ['', '## 两条线各自的边界', '',
              '- 检测器判"有没有板"在真实后厨不可靠，只用于交叉验证与提供食物框；',
              '- VLM 判"有没有板"目前最好，但框常过粗，粗框不会拿去算位置；',
              '- 叠框图：黄色=检测器案板，青色=VLM 案板，红色=检测器食物。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--no-vlm', action='store_true')
    parser.add_argument('--food-confidence', type=float, default=0.25)
    parser.add_argument('--board-confidence', type=float, default=0.5)
    parser.add_argument('--tiling-fallback', choices=['on', 'off'], default='off',
                        help='整帧判无板时用 2x2 分块再判一次（实测可找回圆板，代价 4 次额外推理）')
    parser.add_argument('--cross-check', choices=['on', 'off'], default='off',
                        help='YOLO 案板交叉验证；实测在开发集会误伤 8/10，默认关闭')
    parser.add_argument('--board-box', default=None,
                        help='人工板框，归一化 x1,y1,x2,y2（0-1000）；给了就以它为准，位置仍经过粗框检查')
    parser.add_argument('--board-json', type=Path, default=None,
                        help='从 JSON 读人工板框，格式 {"board": [x1,y1,x2,y2]}')
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error('图片不存在')
    if args.output_dir.exists():
        parser.error('请使用新的输出目录')

    if args.board_box and args.board_json:
        parser.error('--board-box 与 --board-json 只能选一个')
    human_box = None
    if args.board_box:
        human_box = [float(v) for v in args.board_box.split(',')]
        if len(human_box) != 4 or not (0 <= human_box[0] < human_box[2] <= 1000
                                       and 0 <= human_box[1] < human_box[3] <= 1000):
            parser.error('--board-box 需要 4 个 0-1000 的整数且 x1<x2, y1<y2')
    elif args.board_json:
        try:
            human_box = load_human_annotation(args.board_json, args.image)
        except (ValueError, TypeError, OSError) as exc:
            parser.error(str(exc))

    if human_box is not None:
        try:
            validate_human_box(human_box)
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
    detectors = load_detectors()
    vlm = None if args.no_vlm else load_vlm()
    result, image = analyze(args.image, detectors, vlm, no_vlm=args.no_vlm,
                            food_confidence=args.food_confidence,
                            board_confidence=args.board_confidence, draw_overlay=True,
                            cross_check=args.cross_check == 'on', human_box=human_box,
                            tiling_fallback=args.tiling_fallback == 'on')

    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / 'raw_vlm_response.txt').write_text(result['lines']['vlm'].get('raw') or '', encoding='utf-8')
    (args.output_dir / 'review_report.md').write_text(report_text(result), encoding='utf-8')
    if image is not None:
        image.save(args.output_dir / 'review.jpg')

    print(json.dumps({'board_presence': result['board_presence'], 'board_source': result['board_source'],
                      'position_usable': result['position_usable'],
                      'vlm_valid': result['lines']['vlm']['valid'],
                      'yolo_boards': len(result['lines']['detector']['board_boxes']),
                      'food_boxes': len(result['food_from_detector']),
                      'food_locations': [r['location'] for r in result['food_regions']]},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
