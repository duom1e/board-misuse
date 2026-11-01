"""只用一问的案板复核入口：只问"画面里有没有切菜板"，一次推理，输出仍为 needs_review。

为什么单独做：49 帧实测显示，单独问 board 时输出可解析率 49/49，而有板帧命中 2/2；
把容器/食物绑在同一问里会让整条回答更容易失败，反而拖垮案板判断。

用法：
  python review_board_only.py <图片> --output-dir <新目录> [--response-file <保存的原始响应>] [--max-new-tokens 200]
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from region_response_validation import box_quality, validate_region_response

BASE = Path(__file__).resolve().parent
PROMPT_FILE = BASE / 'experiments' / 'split_prompts_v1' / 'prompt_boards.txt'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--response-file', type=Path, help='回放已保存的原始响应，不加载模型')
    parser.add_argument('--max-new-tokens', type=int, default=200)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error('图片不存在')
    if args.output_dir.exists():
        parser.error('请使用新的输出目录，已有结果不会被覆盖')

    prompt = PROMPT_FILE.read_text(encoding='utf-8')
    image = Image.open(args.image).convert('RGB')
    revision = None
    if args.response_file:
        answer = args.response_file.read_text(encoding='utf-8')
    else:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        download = json.loads((BASE / 'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
        revision = download['revision']
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            download['folder'], dtype=torch.bfloat16, device_map='cuda:0',
            attn_implementation='sdpa', local_files_only=True).eval()
        processor = AutoProcessor.from_pretrained(download['folder'], local_files_only=True)
        small = image.copy()
        small.thumbnail((512, 512))
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': small},
                                                {'type': 'text', 'text': prompt}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    # 只取 boards 段；容器/食物在这里恒为空，避免"其它字段失败"污染案板判断
    boards = None
    parse_error = None
    try:
        start, end = answer.find('{'), answer.rfind('}')
        if start < 0 or end < 0:
            raise ValueError('响应里找不到 JSON 对象')
        piece = json.loads(answer[start:end + 1])
        value = piece.get('boards')
        if not isinstance(value, list):
            raise ValueError('缺少 boards 列表')
        boards = value
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)[:300]

    combined = json.dumps({'boards': boards if boards is not None else [], 'containers': [], 'food': []})
    validation = validate_region_response(combined)
    if parse_error:
        validation['valid'] = False
        validation['board_state'] = 'unknown'
        validation['parse_error'] = parse_error

    quality = None
    if validation['board_state'] == 'candidate_present':
        quality = box_quality(validation['normalized_boxes']['boards'])
    result = {
        'status': 'needs_review',
        'compliance_determined': False,
        'image': str(args.image.resolve()),
        'sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'model_revision': revision,
        'prompt': prompt,
        'raw_response': answer,
        'inference_performed': not bool(args.response_file),
        'board_only': True,
        'output_validation': validation,
        'board_state': validation['board_state'],
        'board_position': quality,
        'position_usable': bool(quality and quality['position_usable']),
        'next_action': ('位置可用，仍需人工确认是否接触食材' if quality and quality['position_usable']
                        else '框过粗或为整幅，位置不可用：请人工框定板面范围' if quality
                        else '未报出案板候选：不能作为无板结论'),
        'limitations': '一次推理只回答"有没有切菜板"；矩形框不代表接触，空结果不代表画面无板，仍需人工复核。',
    }
    if parse_error:
        result['parse_error'] = parse_error

    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / 'raw_response.txt').write_text(answer, encoding='utf-8')

    if validation['board_state'] == 'candidate_present' and not parse_error:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(FONT), 18)
        width, height = image.size
        degenerate = {entry['index'] for entry in quality['degenerate_boxes']}
        for index, box in enumerate(validation['normalized_boxes']['boards']):
            x1, y1, x2, y2 = (box[0] * width / 1000, box[1] * height / 1000,
                              box[2] * width / 1000, box[3] * height / 1000)
            if index in degenerate:
                # 整幅框画虚线边框，避免看起来像一个精确的板面框
                step = max(12, width // 40)
                for start in range(0, width, step * 2):
                    draw.line((start, y1, min(start + step, width), y1), fill='orange', width=4)
                    draw.line((start, y2, min(start + step, width), y2), fill='orange', width=4)
                for start in range(0, height, step * 2):
                    draw.line((x1, start, x1, min(start + step, height)), fill='orange', width=4)
                    draw.line((x2, start, x2, min(start + step, height)), fill='orange', width=4)
                draw.text((x1 + 6, y1 + 26), 'board box too coarse - frame it manually',
                          fill='orange', font=font)
            else:
                draw.rectangle((x1, y1, x2, y2), outline='cyan', width=4)
                draw.text((x1 + 3, y1 + 3), f'board {index + 1}', fill='cyan', font=font)
        image.save(args.output_dir / 'review.jpg')

    labels = {'unknown': '未知：输出未通过校验', 'candidate_present': '存在案板候选，需人工复核',
              'no_board_detected': '未检测到案板，不代表画面无板'}
    lines = ['# 案板复核（单问）', '', labels[validation['board_state']], '',
             '状态：needs_review，未判定合规。', '',
             '处理方式：' + ('保存响应回放，未执行本次推理。' if args.response_file else '本次实际执行本地 4B 推理（1 次）。')]
    if quality is not None:
        lines += ['', '## 位置质量', '',
                  f"框数 {quality['boxes']}，可用框 {quality['usable_boxes']}，整幅框 {len(quality['degenerate_boxes'])}",
                  '', f"下一步：{result['next_action']}"]
    if validation['errors']:
        lines += ['', '## 输出错误', '']
        lines += [f"- `{e['path']}`：`{e['code']}` — {e['message']}" for e in validation['errors']]
        lines += ['', '整条结果保持未知，不生成叠框图。']
    elif validation['board_state'] == 'candidate_present' and quality and quality['position_usable']:
        lines += ['', '叠框图为 review.jpg；框正确性、是否接触食材都需人工确认。']
    elif validation['board_state'] == 'candidate_present':
        lines += ['', 'review.jpg 中整幅框以橙色虚线标出并注明 "too coarse"；**不要**据此判断食材是否落在板上。']
    else:
        lines += ['', '本帧没有报出案板候选；这**不能**作为"没有案板"的证明。']
    (args.output_dir / 'validation_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary = f"{args.output_dir.name}: board_state={validation['board_state']} valid={validation['valid']}"
    if quality is not None:
        summary += f" position_usable={quality['position_usable']}"
    if validation['errors']:
        summary += ' codes=' + ','.join(f'{k}x{v}' for k, v in sorted(validation['errors_by_code'].items()))
    print(summary)


if __name__ == '__main__':
    main()
