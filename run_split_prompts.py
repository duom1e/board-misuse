"""方案 B：拆成三个单任务提问（板 / 容器 / 食物），合并后再过同一套校验。

判据先写死（experiments/split_prompts_v1/decision_rule.json）：
  主指标 = 合并后 schema 有效率（49 帧合计）
  支持条件 = 有效率 ≥ 95%
  附加观察 = 无板帧误报数、board_state 相对基线改变比例、单任务失败分布
  警告：如果有效率升了但误报显著增加，算"以多报换通过"，不算支持。
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXP = BASE / 'experiments' / 'split_prompts_v1'
DIRS = [BASE / 'experiments' / 'new_source_video_v1', BASE / 'experiments' / 'new_source_video_v2']
TASKS = [('boards', 'prompt_boards.txt'), ('containers', 'prompt_containers.txt'), ('food', 'prompt_food.txt')]


def main():
    (EXP / 'decision_rule.json').write_text(json.dumps({
        'question': '把三件事拆成三个单任务提问，合并后能否把 schema 有效率提到 95% 以上？',
        'prompts': [f for _, f in TASKS],
        'data': [d.name for d in DIRS],
        'frozen_before_run': True,
        'primary_metric': '合并后 schema 有效率（valid 帧 / 49）',
        'supported_if': '有效率 ≥ 0.95',
        'warning_rule': '若有效率提升但无板误报增加 ≥2 帧，判为"以多报换通过"，不支持',
        'limits': ['有板样本仅 2 帧，不评价召回', '样本为开发集复核', '每帧推理次数从 1 变 3'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    from PIL import Image
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from region_response_validation import validate_region_response

    download = json.loads((BASE / 'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        download['folder'], dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(download['folder'], local_files_only=True)
    prompts = {key: (EXP / name).read_text(encoding='utf-8') for key, name in TASKS}

    all_rows = []
    for directory in DIRS:
        manifest = json.loads((directory / 'frames_manifest.json').read_text(encoding='utf-8'))
        labels = json.loads((directory / 'labels_blind.json').read_text(encoding='utf-8'))['labels']
        baseline_file = directory / 'comparison_v1' / 'results.json'
        baseline = json.loads(baseline_file.read_text(encoding='utf-8')) if baseline_file.exists() else {'rows': []}
        baseline_rows = {row['frame']: row for row in baseline['rows']}

        for record in manifest['frames']:
            name = Path(record['file']).name
            image = Image.open(record['file']).convert('RGB')
            small = image.copy()
            small.thumbnail((512, 512))
            merged = {'boards': [], 'containers': [], 'food': []}
            task_errors = {}
            for key, _ in TASKS:
                messages = [{'role': 'user', 'content': [{'type': 'image', 'image': small},
                                                        {'type': 'text', 'text': prompts[key]}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors='pt').to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                merged[key] = answer
            # 合并前把三段各自当作该字段的响应，再整体校验；未给出 JSON 的字段记空并记录失败任务
            parts = {}
            for key, _ in TASKS:
                text = merged[key].strip()
                parts[key] = text
            combined = '{}'
            counts = {}
            try:
                blocks = {}
                for key, _ in TASKS:
                    text = parts[key]
                    start = text.find('{')
                    end = text.rfind('}')
                    if start < 0 or end < 0:
                        raise ValueError(f'{key}: no json object')
                    piece = json.loads(text[start:end + 1])
                    if key not in piece or not isinstance(piece[key], list):
                        raise ValueError(f'{key}: missing key')
                    blocks[key] = piece[key]
                combined = json.dumps({'boards': blocks['boards'], 'containers': blocks['containers'],
                                       'food': blocks['food']})
                validation = validate_region_response(combined)
            except Exception as exc:  # noqa: BLE001
                task_errors['merge'] = str(exc)[:200]
                validation = validate_region_response('{"boards":[],"containers":[],"food":[]}')
                validation['valid'] = False
                validation['forced_invalid'] = task_errors['merge']

            label = labels.get(name, {})
            predicted = validation['board_state']
            if label.get('usable') is False:
                verdict = 'excluded'
            elif label.get('board_present') == 'yes':
                verdict = 'hit' if predicted == 'candidate_present' else 'miss'
            elif label.get('board_present') == 'no':
                verdict = 'false_alarm' if predicted == 'candidate_present' else 'correct_silent'
            else:
                verdict = 'unclear'
            old = baseline_rows.get(name, {})
            all_rows.append({'frame': name, 'source': directory.name,
                             'label': label, 'board_state': predicted, 'valid': validation['valid'],
                             'errors_by_code': validation['errors_by_code'],
                             'task_errors': task_errors,
                             'baseline_state': old.get('board_state'),
                             'verdict': verdict,
                             'raw_boards': parts['boards'], 'raw_containers': parts['containers'],
                             'raw_food': parts['food']})
            print(name, predicted, verdict, flush=True)

    total = len(all_rows)
    valid = sum(1 for row in all_rows if row['valid'])
    usable = [row for row in all_rows if row['verdict'] != 'excluded']
    summary = {
        'frames': total, 'valid': valid, 'valid_rate': round(valid / total, 4) if total else None,
        'schema_failures': total - valid,
        'frames_usable': len(usable),
        'board_present_labels': sum(1 for row in usable if row['label'].get('board_present') == 'yes'),
        'hits': sum(1 for row in all_rows if row['verdict'] == 'hit'),
        'misses': sum(1 for row in all_rows if row['verdict'] == 'miss'),
        'correct_silent': sum(1 for row in all_rows if row['verdict'] == 'correct_silent'),
        'false_alarms': sum(1 for row in all_rows if row['verdict'] == 'false_alarm'),
        'board_state_changed_vs_baseline': sum(1 for row in all_rows
                                               if row['baseline_state'] and row['baseline_state'] != row['board_state']),
        'note': '对照诊断，不是准确率；每帧推理 3 次',
    }
    out = BASE / 'experiments' / 'split_prompts_v1' / 'results.json'
    out.write_text(json.dumps({'summary': summary, 'rows': all_rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
