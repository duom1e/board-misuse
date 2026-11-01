"""修正 v2 的一处盲标错误（E66 190s 实为白色砧板），并重算受影响的指标。

证据：experiments/disagreement_review/v2_E66_190s_zoom.jpg（放大的白色砧板，带挂孔）
原则：原始盲标保留，追加 revision 字段；所有派生数字重算，不覆盖历史结论而注明更正。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
V2_LABELS = BASE / 'experiments' / 'new_source_video_v2' / 'labels_blind.json'
TARGET = 'BV1ocXNBrE66_190s.jpg'
EVIDENCE = 'experiments/disagreement_review/v2_E66_190s_zoom.jpg'


def fam(row):
    return 'present' if row['board_presence'].startswith('present') else 'absent'


def revise_labels():
    data = json.loads(V2_LABELS.read_text(encoding='utf-8'))
    labels = data['labels']
    entry = labels[TARGET]
    entry['label_original_blind'] = entry['board_present']
    entry['board_present'] = 'yes'
    entry['revision_reason'] = '放大后可见不锈钢台上的白色砧板（带挂孔），此前误判为"无近景操作台"'
    entry['revision_evidence'] = EVIDENCE
    entry['reviewed_after_model_output'] = True
    counts = {'total': len(labels), 'usable': sum(1 for v in labels.values() if v['usable']),
              'board_present': sum(1 for v in labels.values() if v['board_present'] == 'yes'),
              'board_unclear': sum(1 for v in labels.values() if v['board_present'] == 'unclear'),
              'board_absent': sum(1 for v in labels.values() if v['board_present'] == 'no')}
    data['counts'] = counts
    data['revision_log'] = {
        'revised_at': '2026-11-01T23:02:00+08:00',
        'what': '1 帧把白色砧板误标为无板',
        'trigger': '对 v1–v3 做"分歧帧回看"时发现的唯一一处分歧',
        'how': '保留 label_original_blind，追加 revision_reason / revision_evidence',
    }
    V2_LABELS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return counts


def recompute(batch_dir, dirs):
    summary = json.loads((BASE / 'experiments' / batch_dir / 'summary.json').read_text(encoding='utf-8'))
    rows = summary['rows']
    labels = {}
    for name in dirs:
        labels.update(json.loads((BASE / 'experiments' / name / 'labels_blind.json').read_text(encoding='utf-8'))['labels'])
    usable = [r for r in rows if labels.get(r['frame'], {}).get('usable') is not False]
    yes = [r for r in usable if labels[r['frame']]['board_present'] == 'yes']
    no = [r for r in usable if labels[r['frame']]['board_present'] == 'no']
    return {
        'frames_total': len(rows), 'frames_usable': len(usable),
        'board_frames': len(yes), 'board_hits': sum(1 for r in yes if fam(r) == 'present'),
        'no_board_frames': len(no), 'false_alarms': sum(1 for r in no if fam(r) == 'present'),
        'position_usable_among_board': sum(1 for r in yes if r['position_usable']),
    }


def main():
    counts = revise_labels()
    print('v2 labels now:', json.dumps(counts, ensure_ascii=False))
    v1v2 = recompute('combined_batch_v1', ['new_source_video_v1', 'new_source_video_v2'])
    v3 = recompute('combined_batch_v3', ['new_source_video_v3'])
    v4 = json.loads((BASE / 'experiments' / 'combined_batch_v4' / 'recomputed_after_label_fix.json').read_text(encoding='utf-8'))
    total = {
        'board_frames': v1v2['board_frames'] + v3['board_frames'] + v4['board_frames'],
        'board_hits': v1v2['board_hits'] + v3['board_hits'] + v4['board_hits'],
        'no_board_frames': v1v2['no_board_frames'] + v3['no_board_frames'] + v4['no_board_frames'],
        'false_alarms': v1v2['false_alarms'] + v3['false_alarms'] + v4['false_alarms'],
        'position_usable_among_board': (v1v2['position_usable_among_board'] + v3['position_usable_among_board']
                                        + v4['position_usable_among_board_among_board']
                                        if 'position_usable_among_board_among_board' in v4
                                        else v1v2['position_usable_among_board'] + v3['position_usable_among_board']
                                        + v4['position_usable_among_board_frames']),
    }
    out = {'v1v2': v1v2, 'v3': v3, 'v4': v4, 'cumulative_after_corrections': total}
    (BASE / 'experiments' / 'cumulative_board_evidence.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
