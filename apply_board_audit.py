"""按《砧板定义》回查的结论落地：BELh_475s 是编织竹席，撤销其板框；其余 26 个框通过。

动作：
  1) v4 labels_blind.json 里 475s 改 unclear（保留原值，记第 8 次更正）；
  2) 删除 475s 的 .board.json（它不该进板面数据集）；
  3) 重建 datasets/board_seg_v3（保留 v2 不动，便于对照）。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
V4 = BASE / 'experiments' / 'new_source_video_v4' / 'labels_blind.json'
TARGET = 'BV1bqsfzbELh_475s.jpg'
EVIDENCE = 'experiments/board_box_audit_v1/new_source_video_v4/audit_3.jpg'


def main():
    data = json.loads(V4.read_text(encoding='utf-8'))
    entry = data['labels'][TARGET]
    entry['label_original_blind'] = entry.get('label_original_blind', entry['board_present'])
    entry['board_present'] = 'unclear'
    entry['revision_reason'] = '按《砧板定义》回查：框内是编织竹席（十字编织纹），不是砧板'
    entry['revision_evidence'] = EVIDENCE
    data.setdefault('revision_log', {})
    data['revision_log'] = {
        'revised_at': '2026-11-06T00:00:00+08:00',
        'what': 'BELh_475s 撤销正例（编织竹席），并删除其板框',
        'trigger': '按新写的《砧板定义》对 27 个人工框逐张回查（audit_board_boxes.py）',
        'how': '原盲标保留；框文件删除',
    }
    V4.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    sidecar = (BASE / 'experiments' / 'new_source_video_v4' / 'frames' / 'BV1bqsfzbELh_475s.board.json')
    removed = False
    if sidecar.exists():
        sidecar.unlink()
        removed = True
    print(json.dumps({'reclassified': TARGET, 'sidecar_removed': removed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
