"""修正 v4 的三处盲标错误，并重算指标（保留原始盲标，不覆盖）。

触发原因：复核合体入口的"误报"时，把 300s/370s/405s 三帧单独渲染出来看，
发现画面里确实是**黑色砧板（切菜垫）**，只是我之前把深色板误认成了炒锅。
模型判对、我判错。按"先记录再修正"的原则：原始标签保留，新增 revisions。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
LABELS = BASE / 'experiments' / 'new_source_video_v4' / 'labels_blind.json'
EVIDENCE = 'experiments/combined_v4_fp'

REVISIONS = {
    'BV1bqsfzbELh_300s.jpg': {'from': 'no', 'to': 'yes',
                             'reason': '画面下半是黑色砧板（切菜垫）上的蘑菇片与刀，不是金属托盘'},
    'BV1bqsfzbELh_370s.jpg': {'from': 'no', 'to': 'yes',
                             'reason': '黑色砧板（MATX MAT）上切红洋葱，非炒锅'},
    'BV1bqsfzbELh_405s.jpg': {'from': 'no', 'to': 'yes',
                             'reason': '同一块黑色砧板上切洋葱，非炒锅'},
}


def main():
    data = json.loads(LABELS.read_text(encoding='utf-8'))
    labels = data['labels']
    for name, change in REVISIONS.items():
        if name not in labels:
            continue
        labels[name]['label_original_blind'] = labels[name]['board_present']
        labels[name]['board_present'] = change['to']
        labels[name]['revision_reason'] = change['reason']
        labels[name]['revision_evidence'] = f"{EVIDENCE}/fp_{name.split('_')[-1].replace('.jpg', '')}/review.jpg"
        labels[name]['reviewed_after_model_output'] = True

    counts = {'total': len(labels), 'usable': sum(1 for v in labels.values() if v['usable']),
              'board_present': sum(1 for v in labels.values() if v['board_present'] == 'yes'),
              'board_unclear': sum(1 for v in labels.values() if v['board_present'] == 'unclear'),
              'board_absent': sum(1 for v in labels.values() if v['board_present'] == 'no')}
    data['counts'] = counts
    data['revision_log'] = {
        'revised_at': '2026-11-01T22:52:00+08:00',
        'what': '3 帧把黑色砧板误标为无板',
        'why': '复核模型"误报"时逐帧看图，发现画面里确有黑色砧板',
        'how': '保留 label_original_blind，新增 revisions 与证据路径，不删除原始值',
        'impact': 'v4 指标重算：有板从 9 帧变 12 帧，无板从 11 帧变 8 帧',
        'note': '这说明此前各批的单人盲标同样可能有错误；凡"模型与标签不一致"的帧都应逐帧看图复核',
    }
    LABELS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(counts, ensure_ascii=False))

    # 重算 v4 指标
    summary = json.loads((BASE / 'experiments' / 'combined_batch_v4' / 'summary.json').read_text(encoding='utf-8'))
    rows = summary['rows']
    fam = lambda r: 'present' if r['board_presence'].startswith('present') else 'absent'
    yes = [r for r in rows if labels.get(r['frame'], {}).get('board_present') == 'yes'
           and labels.get(r['frame'], {}).get('usable') is not False]
    no = [r for r in rows if labels.get(r['frame'], {}).get('board_present') == 'no'
          and labels.get(r['frame'], {}).get('usable') is not False]
    recomputed = {
        'board_frames': len(yes),
        'board_hits': sum(1 for r in yes if fam(r) == 'present'),
        'no_board_frames': len(no),
        'false_alarms': sum(1 for r in no if fam(r) == 'present'),
        'position_usable_among_board_frames': sum(1 for r in yes if r['position_usable']),
        'vlm_parse_failures': sum(1 for r in rows if not r['vlm_valid']),
    }
    (BASE / 'experiments' / 'combined_batch_v4' / 'recomputed_after_label_fix.json').write_text(
        json.dumps(recomputed, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(recomputed, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
