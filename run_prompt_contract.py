"""提示词契约实验 v1：严格输出契约（限 3 框、只输出 JSON）能否降低 schema 失败率。

判据在跑之前写死（experiments/prompt_contract_v1/decision_rule.json）：
  主指标 = schema 有效帧占比（49 帧合计）
  支持条件 = 有效率 ≥95%，且 board_state 相对基线改变 ≤10%
  注意：这是"输出格式"实验，不评价识别准不准；有板样本仅 2 帧。
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXP = BASE / 'experiments' / 'prompt_contract_v1'
PROMPT_FILE = EXP / 'prompt_constrained.txt'
DIRS = [BASE / 'experiments' / 'new_source_video_v1', BASE / 'experiments' / 'new_source_video_v2']
OUT_SUB = 'comparison_contract_v1'


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / 'decision_rule.json').write_text(json.dumps({
        'question': '把输出契约写死（每类最多 3 框、只输出一个 JSON、给示例）能否降低 schema 失败率？',
        'prompt_file': str(PROMPT_FILE.relative_to(BASE)),
        'baseline': 'comparison_v1（原提示词，每类最多 6 框）',
        'data': [d.name for d in DIRS],
        'frozen_before_run': True,
        'primary_metric': 'schema 有效率 = valid 帧 / 总帧（两批合计 49 帧）',
        'secondary_metrics': ['board_state 相对基线改变比例', 'box_count / json_truncated / json_error 分布'],
        'decision': {
            'supported_if': '合计有效率 ≥95% 且 board_state 改变 ≤10%',
            'not_supported_if': '有效率提升但 board_state 大量改变（说明是靠"少报"换通过率）',
        },
        'limits': ['有板样本仅 2 帧，不能测召回', '同一批帧已被基线看过，属开发集复核', '不据此改默认提示词'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    import evaluate_new_source_v1 as evaluation

    for directory in DIRS:
        sys.argv = ['evaluate_new_source_v1.py', '--dir', str(directory), '--out', OUT_SUB,
                    '--prompt-file', str(PROMPT_FILE)]
        evaluation.main()

    rows = []
    total = valid = 0
    changed = 0
    codes = {}
    for directory in DIRS:
        base = json.loads((directory / 'comparison_v1' / 'results.json').read_text(encoding='utf-8'))
        new = json.loads((directory / OUT_SUB / 'results.json').read_text(encoding='utf-8'))
        base_rows = {r['frame']: r for r in base['rows']}
        for row in new['rows']:
            total += 1
            valid += 1 if row['valid'] else 0
            for code in row['errors_by_code']:
                codes[code] = codes.get(code, 0) + 1
            old = base_rows.get(row['frame'])
            if old and old['board_state'] != row['board_state']:
                changed += 1
        rows.append({'source': directory.name, 'baseline_failures': base['summary']['schema_failures'],
                     'contract_failures': new['summary']['schema_failures']})

    summary = {
        'frames': total, 'valid': valid, 'valid_rate': round(valid / total, 4) if total else None,
        'schema_failures': total - valid, 'error_codes': codes,
        'board_state_changed': changed,
        'changed_rate': round(changed / total, 4) if total else None,
        'per_source': rows,
    }
    (EXP / 'results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
