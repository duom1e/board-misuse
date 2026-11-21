"""把 BV1ux411g7tW 的 20 帧遮挡标注写入 labels_blind.json。

依据：grid_BV1ux411g7tW.jpg。
· 118s 为洋葱特写，画面内无板 → no
· 25s / 211s / 235s / 258s / 281s / 305s 板面完整（其中 211/235s 板在远处台面上）→ none
· 其余为手/刀/食材遮挡但板缘可辨 → mild
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'occlusion_phase0'
LABELS = OUT / 'labels_blind.json'
SOURCE = 'BV1ux411g7tW'
EVIDENCE = 'experiments/occlusion_phase0/grid_BV1ux411g7tW.jpg'

NO_BOARD = {4: '洋葱特写，画面内无板'}
NONE = {0: '空板置于台面，四边完整',
        8: '厨师口播，板在远处台面，完整可见（目标很小）',
        9: '同上',
        10: '板上单片洋葱，无手',
        11: '板上洋葱丁，无手',
        12: '刀与姜末小碗，板面完整'}
MILD = {
    1: '手与刀压在板上',
    2: '双手与大刀，板缘可辨',
    3: '手持洋葱在板上方，板面部分可见',
    5: '手持洋葱片，板面在后',
    6: '双手处理洋葱，板面可见',
    7: '刀切芹菜段，板缘完整',
    13: '双手与青椒，板缘完整',
    14: '手压青椒，板面可见',
    15: '青椒丁与手，板缘可辨',
    16: '刀与姜，手在右侧',
    17: '刀与姜末，手在右缘',
    18: '手与柠檬、刀',
    19: '手与番茄、刀',
}


def main():
    data = json.loads(LABELS.read_text(encoding='utf-8'))
    manifest = json.loads((OUT / 'blind_v2_manifest.json').read_text(encoding='utf-8'))
    frames = [f for f in manifest['frames'] if f['source_bvid'] == SOURCE]
    counts = {}
    for slot, record in enumerate(frames):
        name = Path(record['file']).name
        if slot in NO_BOARD:
            entry = {'board_present': 'no', 'occlusion_level': 'not_applicable',
                     'visible_box': None, 'full_box': None, 'note': NO_BOARD[slot]}
        elif slot in NONE:
            entry = {'board_present': 'yes', 'occlusion_level': 'none',
                     'visible_box': None, 'full_box': None, 'note': NONE[slot]}
        else:
            entry = {'board_present': 'yes', 'occlusion_level': 'mild',
                     'visible_box': None, 'full_box': None,
                     'occluder': ['hand', 'knife', 'vegetable'], 'note': MILD[slot]}
        entry.update({'label_confidence': 'high', 'evidence': EVIDENCE,
                      'source_bvid': SOURCE, 'seconds': record['seconds']})
        data['labels'][name] = entry
        key = 'no' if entry['board_present'] == 'no' else entry['occlusion_level']
        counts[key] = counts.get(key, 0) + 1
    data.setdefault('batch_log', []).append(
        {'source': SOURCE, 'frames': len(frames), 'counts': counts,
         'evidence': EVIDENCE, 'at': '2026-11-21T16:35:00+08:00'})
    data['counts'] = {'labelled_total': len(data['labels'])}
    LABELS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(SOURCE, counts, 'total', len(data['labels']))


if __name__ == '__main__':
    main()
