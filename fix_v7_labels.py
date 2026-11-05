"""v7 标注复核：045s/170s 的"案板"其实是**编织竹席/垫子**，按 unclear 处理；220s 是白盘，属误报。

保留原始盲标（label_original_blind），并重算 v7 与累计证据。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
LABELS = BASE / 'experiments' / 'new_source_video_v7' / 'labels_blind.json'
EVIDENCE = 'experiments/label_verify_v1/v7_review.jpg'
RECLASS = {
    'BV1PW411F7mD_045s.jpg': '画面是编织竹席/垫子，不是标准砧板；判"无板"可辩护',
    'BV1PW411F7mD_170s.jpg': '同一块编织垫，同上',
}


def fam(row):
    return 'present' if row['board_presence'].startswith('present') else 'absent'


def main():
    data = json.loads(LABELS.read_text(encoding='utf-8'))
    labels = data['labels']
    for name, reason in RECLASS.items():
        entry = labels[name]
        entry['label_original_blind'] = entry['board_present']
        entry['board_present'] = 'unclear'
        entry['revision_reason'] = reason
        entry['revision_evidence'] = EVIDENCE
    data['counts'] = {
        'total': len(labels),
        'usable': sum(1 for v in labels.values() if v['usable']),
        'board_present': sum(1 for v in labels.values() if v['board_present'] == 'yes'),
        'board_unclear': sum(1 for v in labels.values() if v['board_present'] == 'unclear'),
        'board_absent': sum(1 for v in labels.values() if v['board_present'] == 'no'),
    }
    data['revision_log'] = {
        'revised_at': '2026-11-05T23:50:00+08:00',
        'what': '2 帧"木砧板"实为编织竹席，改标 unclear（不计入正负例）',
        'why': '复核模型"漏检"时放大确认：表面是编织纹垫子，不是标准砧板',
        'keep': '原始盲标保留在 label_original_blind',
    }
    LABELS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    rows = json.loads((BASE / 'experiments' / 'combined_batch_v7' / 'summary.json').read_text(encoding='utf-8'))['rows']
    yes = [r for r in rows if labels[r['frame']]['board_present'] == 'yes' and labels[r['frame']]['usable']]
    no = [r for r in rows if labels[r['frame']]['board_present'] == 'no' and labels[r['frame']]['usable']]
    v7 = {
        'frames': len(rows),
        'board_frames': len(yes),
        'board_hits': sum(1 for r in yes if fam(r) == 'present'),
        'position_usable_among_board': sum(1 for r in yes if r['position_usable']),
        'no_board_frames': len(no),
        'false_alarms': sum(1 for r in no if fam(r) == 'present'),
        'vlm_parse_failures': sum(1 for r in rows if not r['vlm_valid']),
    }
    cumulative_file = BASE / 'experiments' / 'cumulative_board_evidence.json'
    cumulative = json.loads(cumulative_file.read_text(encoding='utf-8'))
    old = cumulative['cumulative_after_corrections']
    cumulative['v7'] = v7
    cumulative['cumulative_after_corrections'] = {
        'board_frames': old['board_frames'] + v7['board_frames'],
        'board_hits': old['board_hits'] + v7['board_hits'],
        'no_board_frames': old['no_board_frames'] + v7['no_board_frames'],
        'false_alarms': old['false_alarms'] + v7['false_alarms'],
        'position_usable_among_board': old['position_usable_among_board'] + v7['position_usable_among_board'],
    }
    cumulative['revision_count'] = 7
    cumulative_file.write_text(json.dumps(cumulative, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'v7': v7, 'cumulative': cumulative['cumulative_after_corrections']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
