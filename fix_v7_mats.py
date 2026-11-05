"""v7 重大更正：那 6 帧"木砧板"其实都是**编织竹席**，按定义不算板。

这意味着 v7 的"6/6 命中"是错的：模型在竹席上判了"有板"，那不是命中，而是误报。
本脚本：
  1) 把 v7 全部 8 帧竹席画面标为 board_present='no'（附原因与证据），原始值保留；
  2) 重算 v7 与累计证据：有板帧减少 6，无板帧增加 8，误报增加 6。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
V7 = BASE / 'experiments' / 'new_source_video_v7' / 'labels_blind.json'
EVIDENCE = 'experiments/label_verify_v1/v7_mat_check.jpg'
MAT_FRAMES = ['BV1PW411F7mD_020s.jpg', 'BV1PW411F7mD_045s.jpg', 'BV1PW411F7mD_070s.jpg',
              'BV1PW411F7mD_095s.jpg', 'BV1PW411F7mD_120s.jpg', 'BV1PW411F7mD_145s.jpg',
              'BV1PW411F7mD_170s.jpg', 'BV1PW411F7mD_195s.jpg']


def fam(row):
    return 'present' if row['board_presence'].startswith('present') else 'absent'


def main():
    data = json.loads(V7.read_text(encoding='utf-8'))
    labels = data['labels']
    for name in MAT_FRAMES:
        entry = labels[name]
        entry['label_original_blind'] = entry.get('label_original_blind', entry['board_present'])
        entry['board_present'] = 'no'
        entry['revision_reason'] = ('放大确认是编织竹席（十字编织纹），按《砧板定义》不算砧板；'
                                    '模型判"有板"属于误报，不是命中')
        entry['revision_evidence'] = EVIDENCE
    data['counts'] = {
        'total': len(labels),
        'usable': sum(1 for v in labels.values() if v['usable']),
        'board_present': sum(1 for v in labels.values() if v['board_present'] == 'yes'),
        'board_unclear': sum(1 for v in labels.values() if v['board_present'] == 'unclear'),
        'board_absent': sum(1 for v in labels.values() if v['board_present'] == 'no'),
    }
    data['revision_log'] = {
        'revised_at': '2026-11-06T00:10:00+08:00',
        'what': '8 帧编织竹席全部改判"非板"；撤销 v7 的"6/6 命中"结论',
        'trigger': '准备给这批打框时先看材质，发现整段都是同一块竹席',
        'impact': '有板帧 -6、无板帧 +8、误报 +6',
    }
    V7.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

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
    cum = json.loads(cumulative_file.read_text(encoding='utf-8'))
    old = cum['cumulative_after_corrections']
    # v7 之前贡献的是 board 6 / hits 6 / no_board 11 / FA 1；现在改为 board 0 / hits 0 / no_board 19 / FA 7
    prev = cum.get('v7', {'board_frames': 6, 'board_hits': 6, 'no_board_frames': 11, 'false_alarms': 1,
                          'position_usable_among_board': 2})
    cum['v7'] = v7
    cum['cumulative_after_corrections'] = {
        'board_frames': old['board_frames'] - prev['board_frames'] + v7['board_frames'],
        'board_hits': old['board_hits'] - prev['board_hits'] + v7['board_hits'],
        'no_board_frames': old['no_board_frames'] - prev['no_board_frames'] + v7['no_board_frames'],
        'false_alarms': old['false_alarms'] - prev['false_alarms'] + v7['false_alarms'],
        'position_usable_among_board': (old['position_usable_among_board'] - prev['position_usable_among_board']
                                        + v7['position_usable_among_board']),
    }
    cum['revision_count'] = 9
    cumulative_file.write_text(json.dumps(cum, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'v7': v7, 'cumulative': cum['cumulative_after_corrections']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
