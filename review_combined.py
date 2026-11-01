"""合体入口：一条命令同时跑"检测器线"和"VLM 线"，按先锁的规则融合成一份复核结果。

两条线各自擅长什么（2026-11-01 的实测结论）：
  · 检测器线（YOLO）：食物是什么（meat/vegetable）——它判断"有没有板"在真实后厨不可靠；
  · VLM 线（Qwen3-VL-4B 单问）：有没有板——49/49 可解析、2/2 命中，但框常常过粗；
融合规则固定（未事后调整）：

  1. 板存在性：以 VLM 为准；VLM 说"没有"但检测器案板模型 conf≥0.5 检出 → board_uncertain。
  2. 板位置：优先 VLM 框且必须过 box_quality；过粗时退到检测器框；都不可用 → position_usable=false。
  3. 食物：只用检测器框（conf≥0.25），位置归属走 food_region_gate，并入 position_usable 护栏。
  4. 结果永远 needs_review；每条证据标注来自哪条线。

用法：
  python review_combined.py <图片> --output-dir <新目录> [--no-vlm]
批量（同进程加载一次模型）：
  python run_combined_batch.py --hours 0
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from food_region_gate import locate
from region_response_validation import box_quality, validate_region_response

BASE = Path(__file__).resolve().parent
BOARD_PROMPT = BASE / 'experiments' / 'split_prompts_v1' / 'prompt_boards.txt'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')


def load_detectors():
    from ultralytics import YOLO
    registry = json.loads((BASE / 'active_models.json').read_text(encoding='utf-8'))
    return {'board': YOLO(registry['board']['weights']), 'food': YOLO(registry['foodseg']['weights']),
            'registry': registry}


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


def fuse(vlm_state, vlm_quality, vlm_boxes_px, yolo_boards, cross_check=False):
    """固定融合规则；返回 (board_presence, chosen_boxes, chosen_source)。

    cross_check 默认关闭：2026-11-01 在 49 帧开发集上实测，YOLO 案板交叉验证会把
    8 个"明确无板"的帧翻成 uncertain，误伤远大于收益（见 experiments/combined_batch_v1/）。
    """
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


def analyze(image_path, detectors, vlm=None, no_vlm=False, food_confidence=0.25, board_confidence=0.5,
            draw_overlay=False, cross_check=False, human_box=None):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    # 人工框（归一化 0..1000）：有它就以它为准，不再让模型决定板在哪
    human_box_px = None
    if human_box:
        human_box_px = [human_box[0] * width / 1000, human_box[1] * height / 1000,
                        human_box[2] * width / 1000, human_box[3] * height / 1000]

    board_result = detectors['board'].predict(source=str(image_path), conf=board_confidence,
                                              imgsz=960, device=0, verbose=False)[0]
    food_result = detectors['food'].predict(source=str(image_path), conf=food_confidence,
                                            imgsz=960, device=0, verbose=False)[0]
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

    if human_box_px:
        # 人工框优先：存在性、位置都以人工为准，模型只作为交叉参考
        board_presence, chosen_boxes, chosen_source = 'present_human_box', [human_box_px], 'human'
        chosen_quality = box_quality([human_box_px])
        board_position_usable = True
    else:
        board_presence, chosen_boxes, chosen_source = fuse(
            vlm_info['board_state'] if vlm_info['performed'] else None,
            vlm_quality, vlm_boxes_px, yolo_boards, cross_check=cross_check)
        chosen_quality = box_quality(chosen_boxes) if chosen_boxes else {'position_usable': False,
                                                                         'reason': 'no_boxes',
                                                                         'degenerate_boxes': []}
        board_position_usable = (chosen_source in ('vlm', 'yolo', 'yolo_crosscheck')
                                 and chosen_quality['position_usable'])

    food_regions = [{'category': item['category'], 'confidence': item['confidence'], 'box': item['box'],
                     **locate(item['box'], chosen_boxes, [], board_position_usable=board_position_usable)}
                    for item in yolo_food]

    result = {
        'status': 'needs_review',
        'compliance_determined': False,
        'image': str(Path(image_path).resolve()),
        'sha256': hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        'board_presence': board_presence,
        'board_boxes': chosen_boxes,
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
        'fusion': {'rules': 'vlm_presence_priority; vlm_box_if_tight; yolo_box_fallback; needs_review_always',
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
             '', '状态：needs_review，未判定合规。', '',
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
    parser.add_argument('--cross-check', choices=['on', 'off'], default='off',
                        help='YOLO 案板交叉验证；实测在开发集会误伤 8/10，默认关闭')
    parser.add_argument('--board-box', default=None,
                        help='人工板框，归一化 x1,y1,x2,y2（0-1000）；给了就以它为准，位置直接视为可用')
    parser.add_argument('--board-json', type=Path, default=None,
                        help='从 JSON 读人工板框，格式 {"board": [x1,y1,x2,y2]}')
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error('图片不存在')
    if args.output_dir.exists():
        parser.error('请使用新的输出目录')

    human_box = None
    if args.board_box:
        human_box = [float(v) for v in args.board_box.split(',')]
        if len(human_box) != 4 or not (0 <= human_box[0] < human_box[2] <= 1000
                                       and 0 <= human_box[1] < human_box[3] <= 1000):
            parser.error('--board-box 需要 4 个 0-1000 的整数且 x1<x2, y1<y2')
    elif args.board_json:
        payload = json.loads(args.board_json.read_text(encoding='utf-8'))
        human_box = payload.get('board') or payload.get('board_box')
        if not human_box or len(human_box) != 4:
            parser.error('JSON 里需要 {"board": [x1,y1,x2,y2]}')

    detectors = load_detectors()
    vlm = None if args.no_vlm else load_vlm()
    result, image = analyze(args.image, detectors, vlm, no_vlm=args.no_vlm,
                            food_confidence=args.food_confidence,
                            board_confidence=args.board_confidence, draw_overlay=True,
                            cross_check=args.cross_check == 'on', human_box=human_box)

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
