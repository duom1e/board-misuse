"""v5 第五处盲标更正（Eyw 870s 实为绿色砧板）+ 重算 v5 与累计证据。"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
V5 = BASE / 'experiments' / 'new_source_video_v5' / 'labels_blind.json'
TARGET = 'BV18pRhBgEyw_870s.jpg'
EVIDENCE = 'experiments/combined_v5_fp/fp_870s_zoom.jpg'


def fam(row):
    return 'present' if row['board_presence'].startswith('present') else 'absent'


def main():
    data = json.loads(V5.read_text(encoding='utf-8'))
    labels = data['labels']
    entry = labels[TARGET]
    entry['label_original_blind'] = entry['board_present']
    entry['board_present'] = 'yes'
    entry['revision_reason'] = '放大后可见不锈钢台上的绿色砧板（带绿色包边），此前漏看'
    entry['revision_evidence'] = EVIDENCE
    entry['reviewed_after_model_output'] = True
    counts = {'total': len(labels), 'usable': sum(1 for v in labels.values() if v['usable']),
              'board_present': sum(1 for v in labels.values() if v['board_present'] == 'yes'),
              'board_unclear': sum(1 for v in labels.values() if v['board_present'] == 'unclear'),
              'board_absent': sum(1 for v in labels.values() if v['board_present'] == 'no')}
    data['counts'] = counts
    data['revision_log'] = {
        'revised_at': '2026-11-01T23:55:00+08:00',
        'what': '1 帧漏看绿色砧板',
        'trigger': '复核 v5 唯一"误报"时放大查看',
        'this_is': '同类错误的第 5 次（前 4 次在 v2/v4）',
    }
    V5.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    rows = json.loads((BASE / 'experiments' / 'combined_batch_v5' / 'summary.json').read_text(encoding='utf-8'))['rows']
    yes = [r for r in rows if labels[r['frame']]['board_present'] == 'yes' and labels[r['frame']].get('usable') is not False]
    no = [r for r in rows if labels[r['frame']]['board_present'] == 'no' and labels[r['frame']].get('usable') is not False]
    v5 = {
        'frames': len(rows),
        'board_frames': len(yes), 'board_hits': sum(1 for r in yes if fam(r) == 'present'),
        'position_usable_among_board': sum(1 for r in yes if r['position_usable']),
        'no_board_frames': len(no), 'false_alarms': sum(1 for r in no if fam(r) == 'present'),
        'vlm_parse_failures': sum(1 for r in rows if not r['vlm_valid']),
    }
    cumulative_file = BASE / 'experiments' / 'cumulative_board_evidence.json'
    cumulative = json.loads(cumulative_file.read_text(encoding='utf-8'))
    old = cumulative['cumulative_after_corrections']
    cumulative['v5'] = v5
    cumulative['cumulative_after_corrections'] = {
        'board_frames': old['board_frames'] + v5['board_frames'],
        'board_hits': old['board_hits'] + v5['board_hits'],
        'no_board_frames': old['no_board_frames'] + v5['no_board_frames'],
        'false_alarms': old['false_alarms'] + v5['false_alarms'],
        'position_usable_among_board': old['position_usable_among_board'] + v5['position_usable_among_board'],
    }
    cumulative['revision_count'] = 5
    cumulative_file.write_text(json.dumps(cumulative, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'v5': v5, 'cumulative': cumulative['cumulative_after_corrections']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
