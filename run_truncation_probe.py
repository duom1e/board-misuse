"""截断探针 v1：同一批冻结帧、同一提示词，只把 max_new_tokens 从 300 提到 600。

判据（跑之前写死，见 experiments/truncation_probe_v1/decision_rule.json）：
  主指标 = schema 有效帧占比；次指标 = board_state 与 300 版本相比是否改变。
  不比较"模型准不准"——这批帧的有板样本只有 2 帧，本来就不足以测召回。
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'truncation_probe_v1'
DIRS = [BASE / 'experiments' / 'new_source_video_v1', BASE / 'experiments' / 'new_source_video_v2']
TOKENS = 600


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'decision_rule.json').write_text(json.dumps({
        'question': 'max_new_tokens=300 是否导致大量 schema 失败？提高到 600 能否显著降低失败率？',
        'data': [str(d.name) for d in DIRS],
        'frozen_before_run': True,
        'primary_metric': 'schema 有效帧占比（valid=True / 总帧）',
        'secondary_metric': 'board_state 相对 300 版本的变化比例',
        'decision': {
            'supported_if': '有效帧占比提升 ≥15 个百分点，且 board_state 改变比例 ≤15%',
            'note': '即使支持，也只是"输出格式更稳"，不代表识别更准；不据此改任何阈值或提示词'
        },
        'limits': ['有板样本仅 2 帧，不能测召回', '同一批帧已被 300 版本看过，属开发集复核'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    import evaluate_new_source_v1 as evaluation

    results = {}
    for directory in DIRS:
        sys.argv = ['evaluate_new_source_v1.py', '--dir', str(directory),
                    '--out', f'comparison_tokens{TOKENS}', '--max-new-tokens', str(TOKENS)]
        evaluation.main()
        collected = json.loads((directory / f'comparison_tokens{TOKENS}' / 'results.json').read_text(encoding='utf-8'))
        results[directory.name] = collected['summary']

    baseline = {}
    for directory in DIRS:
        base_file = directory / 'comparison_v1' / 'results.json'
        if base_file.exists():
            baseline[directory.name] = json.loads(base_file.read_text(encoding='utf-8'))

    comparison = {}
    for name, summary in results.items():
        old = baseline.get(name, {}).get('summary', {})
        old_rows = {r['frame']: r for r in baseline.get(name, {}).get('rows', [])}
        new_rows = {r['frame']: r for r in json.loads(
            (BASE / 'experiments' / name / f'comparison_tokens{TOKENS}' / 'results.json').read_text(encoding='utf-8'))['rows']}
        changed = sum(1 for frame, row in new_rows.items()
                      if frame in old_rows and old_rows[frame]['board_state'] != row['board_state'])
        comparison[name] = {
            'frames': summary['frames_total'],
            'valid_300': old.get('frames_total', 0) - old.get('schema_failures', 0),
            'schema_failures_300': old.get('schema_failures'),
            'schema_failures_600': summary['schema_failures'],
            'board_state_changed': changed,
        }
    (OUT / 'comparison.json').write_text(json.dumps({'tokens': TOKENS, 'per_source': comparison},
                                                    ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
