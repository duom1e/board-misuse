"""新来源视频对照：加载一次本地 4B，逐帧跑，和人工盲标比较（等 GPU 空闲再跑）。

只用冻结的 24 帧与其 labels_blind.json；不调提示词、不调阈值、不改默认模型。
输出 experiments/new_source_video_v1/comparison_v1/：每帧 raw 响应、校验结果、与标签的对照表。
"""
import argparse
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_EXP = BASE / 'experiments' / 'new_source_video_v1'
EXP = DEFAULT_EXP
OUT = DEFAULT_EXP / 'comparison_v1'


def main():
    global EXP, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=str(DEFAULT_EXP), help='实验目录（含 frames_manifest.json 与 labels_blind.json）')
    parser.add_argument('--out', default=None, help='输出子目录名，默认 comparison_v1')
    parser.add_argument('--prompt-file', default=None, help='改用自定义提示词（新实验，需另锁计划）')
    parser.add_argument('--max-new-tokens', type=int, default=300, help='生成上限；改它是新实验，需另锁计划')
    args = parser.parse_args()
    EXP = Path(args.dir)
    OUT = EXP / (args.out or 'comparison_v1')

    from PIL import Image
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.prompt_file:
        PROMPT = Path(args.prompt_file).read_text(encoding='utf-8')
    else:
        from evaluate_qwen4b_regions import PROMPT
    from region_response_validation import validate_region_response

    manifest = json.loads((EXP / 'frames_manifest.json').read_text(encoding='utf-8'))
    labels = json.loads((EXP / 'labels_blind.json').read_text(encoding='utf-8'))['labels']
    download = json.loads((BASE / 'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        download['folder'], dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(download['folder'], local_files_only=True)

    rows = []
    for record in manifest['frames']:
        name = Path(record['file']).name
        image = Image.open(record['file']).convert('RGB')
        small = image.copy()
        small.thumbnail((512, 512))
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': small},
                                                 {'type': 'text', 'text': PROMPT}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt').to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        validation = validate_region_response(answer)
        label = labels[name]
        predicted = validation['board_state']
        if label['usable'] is False:
            verdict = 'excluded'
        elif label['board_present'] == 'yes':
            verdict = 'hit' if predicted == 'candidate_present' else 'miss'
        elif label['board_present'] == 'no':
            verdict = 'false_alarm' if predicted == 'candidate_present' else 'correct_silent'
        else:
            verdict = 'unclear'
        rows.append({'frame': name, 'source': record['source_bvid'], 'seconds': record['seconds'],
                     'label': label, 'board_state': predicted, 'valid': validation['valid'],
                     'errors_by_code': validation['errors_by_code'],
                     'raw_sha256': validation['raw_sha256'], 'seconds_inference': round(time.perf_counter() - started, 2),
                     'verdict': verdict, 'raw_response': answer})
        print(name, predicted, verdict, flush=True)

    usable = [row for row in rows if row['verdict'] != 'excluded']
    summary = {
        'scope': 'new-source video frames, labels recorded before any model inference',
        'frames_total': len(rows), 'frames_usable': len(usable),
        'board_present_labels': sum(1 for row in usable if row['label']['board_present'] == 'yes'),
        'board_absent_labels': sum(1 for row in usable if row['label']['board_present'] == 'no'),
        'board_unclear_labels': sum(1 for row in usable if row['label']['board_present'] == 'unclear'),
        'hits': sum(1 for row in rows if row['verdict'] == 'hit'),
        'misses': sum(1 for row in rows if row['verdict'] == 'miss'),
        'correct_silent': sum(1 for row in rows if row['verdict'] == 'correct_silent'),
        'false_alarms': sum(1 for row in rows if row['verdict'] == 'false_alarm'),
        'schema_failures': sum(1 for row in rows if not row['valid']),
        'note': '对照诊断，不是准确率；样本量小且只有 1 帧有明确案板，不能外推',
    }
    (OUT / 'results.json').write_text(json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
