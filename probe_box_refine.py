"""框细化探针：VLM 给出整幅/过粗框时，裁到该框再问一次，能否得到更紧的板框？

背景：48 帧有板样本里只有 16 帧位置可用（33%），主要因为特写时 VLM 框覆盖整幅。
判据先写死（见 decision_rule.json）：
  主指标 = 原本"过粗"的有板帧中，细化后位置变为可用的比例（≥50% 视为有效）
  护栏   = 细化不得改变"有/无板"的判断（原本有板，细化后仍须有板）

只做推理，不训练、不改默认。
"""
import json
from pathlib import Path

from PIL import Image

from region_response_validation import box_quality

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'box_refine_probe_v1'
PROMPT = BASE / 'experiments' / 'split_prompts_v1' / 'prompt_boards.txt'
FRAMES = [
    ('new_source_video_v8', 'BV14W41197Rd_020s.jpg'),
    ('new_source_video_v8', 'BV14W41197Rd_060s.jpg'),
    ('new_source_video_v8', 'BV14W41197Rd_100s.jpg'),
    ('new_source_video_v8', 'BV14W41197Rd_140s.jpg'),
    ('new_source_video_v8', 'BV14W41197Rd_180s.jpg'),
    ('new_source_video_v8', 'BV14W41197Rd_220s.jpg'),
    ('new_source_video_v4', 'BV1bqsfzbELh_300s.jpg'),
    ('new_source_video_v4', 'BV1bqsfzbELh_370s.jpg'),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'decision_rule.json').write_text(json.dumps({
        'question': '把 VLM 的过粗框裁出来再问一次，能否把"位置可用"的比例提上来？',
        'frames': len(FRAMES),
        'primary': '原本过粗帧中，细化后 position_usable 变 true 的比例（≥0.5 视为有效）',
        'guard': '细化后仍须判"有板"（不得因细化丢板）',
        'frozen_before_run': True,
        'not_doing': ['不训练', '不改提示词', '不改默认阈值'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from review_combined import parse_vlm_boards

    download = json.loads((BASE / 'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        download['folder'], dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(download['folder'], local_files_only=True)
    prompt = PROMPT.read_text(encoding='utf-8')

    def ask(piece):
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': piece},
                                                {'type': 'text', 'text': prompt}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        return processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    rows = []
    improved = 0
    kept = 0
    for folder, name in FRAMES:
        path = BASE / 'experiments' / folder / 'frames' / name
        image = Image.open(path).convert('RGB')
        width, height = image.size
        small = image.copy(); small.thumbnail((512, 512))
        boxes, _ = parse_vlm_boards(ask(small))
        base_boxes = boxes or []
        base_quality = box_quality(base_boxes) if base_boxes else None
        base_usable = bool(base_quality and base_quality['position_usable'])
        refined_box = None
        refined_usable = False
        if base_boxes and not base_usable:
            # 裁到最大框（含 10% 余量），放大后重问
            biggest = max(base_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            mx = (biggest[2] - biggest[0]) * 0.1
            my = (biggest[3] - biggest[1]) * 0.1
            x1 = max(0, int((biggest[0] - mx) * width / 1000))
            y1 = max(0, int((biggest[1] - my) * height / 1000))
            x2 = min(width, int((biggest[2] + mx) * width / 1000))
            y2 = min(height, int((biggest[3] + my) * height / 1000))
            if x2 - x1 > 32 and y2 - y1 > 32:
                crop = image.crop((x1, y1, x2, y2))
                scale = 512 / max(crop.width, crop.height, 1)
                crop = crop.resize((max(32, int(crop.width * scale)), max(32, int(crop.height * scale))), Image.LANCZOS)
                sub_boxes, _ = parse_vlm_boards(ask(crop))
                if sub_boxes:
                    sub = sub_boxes[0]
                    refined_box = [x1 + (x2 - x1) * sub[0] / 1000, y1 + (y2 - y1) * sub[1] / 1000,
                                   x1 + (x2 - x1) * sub[2] / 1000, y1 + (y2 - y1) * sub[3] / 1000]
                    refined_usable = box_quality([refined_box])['position_usable']
        if refined_usable:
            improved += 1
        if refined_box is not None:
            kept += 1
        rows.append({'frame': name, 'base_boxes': base_boxes, 'base_usable': base_usable,
                     'base_area_fraction': (base_quality['degenerate_boxes'][0]['area_fraction']
                                            if base_quality and base_quality['degenerate_boxes'] else None),
                     'refined_box': refined_box, 'refined_usable': refined_usable})
        print(name, 'base_usable', base_usable, '-> refined_usable', refined_usable, flush=True)

    coarses = [r for r in rows if not r['base_usable']]
    summary = {
        'frames': len(rows),
        'base_usable': sum(1 for r in rows if r['base_usable']),
        'coarse_frames': len(coarses),
        'coarse_refined_usable': improved,
        'improvement_rate': round(improved / len(coarses), 3) if coarses else None,
        'refinement_returned_board': kept,
        'rule_met': (coarses and improved / len(coarses) >= 0.5),
        'note': '单批探针，不是准确率；样本仅 8 帧',
    }
    (OUT / 'results.json').write_text(json.dumps({'summary': summary, 'rows': rows},
                                                 ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
