"""VLM 分块探针：整帧判不出"深色圆板"时，2×2 分块放大能否找回？

判据先写死（experiments/vlm_tiling_probe_v1/decision_rule.json）：
  正例（3 张圆板图，已有近似框）要求：分块后至少 2/3 张能报出板；
  负例（3 张无板图）要求：分块后 0 张误报。
两者同时满足才认为"分块是可用的补救手段"；只满足一半就是有代价的权衡，不算通过。

不做训练、不改提示词、不改默认模型。
"""
import json
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'vlm_tiling_probe_v1'
PROMPT_FILE = BASE / 'experiments' / 'split_prompts_v1' / 'prompt_boards.txt'
POSITIVES = [
    BASE / 'operation_expansion_screen_v2' / 'BV1dHckzwEBA_06.jpg',
    BASE / 'operation_expansion_screen_v2' / 'BV1dHckzwEBA_10.jpg',
    BASE / 'operation_expansion_screen_v2' / 'BV1dHckzwEBA_11.jpg',
]
NEGATIVES = [
    BASE / 'experiments' / 'new_source_video_v5' / 'frames' / 'BV18pRhBgEyw_330s.jpg',
    BASE / 'experiments' / 'new_source_video_v5' / 'frames' / 'BV18pRhBgEyw_600s.jpg',
    BASE / 'experiments' / 'new_source_video_v5' / 'frames' / 'BV1pu411U7Zh_215s.jpg',
]
CROP = 512


def tiles(image, overlap=0.1):
    width, height = image.size
    result = []
    for row in (0, 1):
        for col in (0, 1):
            x1 = int(width * (col * 0.5 - (overlap if col else 0)))
            y1 = int(height * (row * 0.5 - (overlap if row else 0)))
            x2 = int(width * (col * 0.5 + 0.5 + (overlap if not col else 0)))
            y2 = int(height * (row * 0.5 + 0.5 + (overlap if not row else 0)))
            box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            piece = image.crop(box)
            if piece.width < 32 or piece.height < 32:
                continue
            result.append({'box': box, 'image': piece.resize((CROP, CROP), Image.LANCZOS)})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'decision_rule.json').write_text(json.dumps({
        'question': '深色圆板在整帧判不出时，2×2 分块放大能否找回，且不增加负例误报？',
        'positives': [p.name for p in POSITIVES], 'negatives': [n.name for n in NEGATIVES],
        'rule': {'positives_recovered': '>= 2/3', 'negative_false_alarms': '== 0/3'},
        'frozen_before_run': True,
        'not_doing': ['不训练', '不改提示词', '不改默认模型', '不为单图调参'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from region_response_validation import validate_region_response

    download = json.loads((BASE / 'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        download['folder'], dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(download['folder'], local_files_only=True)
    prompt = PROMPT_FILE.read_text(encoding='utf-8')

    def ask(piece):
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': piece},
                                                {'type': 'text', 'text': prompt}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        start, end = answer.find('{'), answer.rfind('}')
        boxes = None
        if start >= 0 and end >= 0:
            try:
                piece_json = json.loads(answer[start:end + 1])
                if isinstance(piece_json.get('boards'), list):
                    boxes = piece_json['boards']
            except Exception:  # noqa: BLE001
                boxes = None
        validation = validate_region_response(json.dumps({'boards': boxes or [], 'containers': [], 'food': []}))
        return {'boxes': boxes, 'state': validation['board_state'], 'raw': answer}

    rows = []
    for kind, items in (('positive', POSITIVES), ('negative', NEGATIVES)):
        for path in items:
            image = Image.open(path).convert('RGB')
            full = image.copy()
            full.thumbnail((CROP, CROP))
            full_result = ask(full)
            tile_results = []
            for tile in tiles(image):
                outcome = ask(tile['image'])
                tile_results.append({'box': tile['box'], 'state': outcome['state'],
                                     'boxes': outcome['boxes']})
            found = any(tile['state'] == 'candidate_present' for tile in tile_results)
            rows.append({'kind': kind, 'image': path.name,
                         'full_frame_state': full_result['state'],
                         'full_frame_boxes': full_result['boxes'],
                         'tiles_board_claimed': sum(1 for tile in tile_results if tile['state'] == 'candidate_present'),
                         'tiling_found_board': found,
                         'tiles': tile_results})
            print(f"{kind:8s} {path.name:24s} full={full_result['state']:18s} tile_hits={rows[-1]['tiles_board_claimed']}", flush=True)

    positives = [r for r in rows if r['kind'] == 'positive']
    negatives = [r for r in rows if r['kind'] == 'negative']
    summary = {
        'positives_total': len(positives),
        'positives_recovered_by_tiling': sum(1 for r in positives if r['tiling_found_board']),
        'positives_full_frame': sum(1 for r in positives if r['full_frame_state'] == 'candidate_present'),
        'negatives_total': len(negatives),
        'negative_false_alarms_by_tiling': sum(1 for r in negatives if r['tiling_found_board']),
        'negatives_full_frame': sum(1 for r in negatives if r['full_frame_state'] == 'candidate_present'),
        'rule_met': (sum(1 for r in positives if r['tiling_found_board']) >= 2
                     and sum(1 for r in negatives if r['tiling_found_board']) == 0),
    }
    (OUT / 'results.json').write_text(json.dumps({'summary': summary, 'rows': rows},
                                                 ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
