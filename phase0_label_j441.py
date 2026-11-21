"""把 BV1j441157px 的 20 帧遮挡标注写入 labels_blind.json。

依据：grid_BV1j441157px.jpg。
· 139/161/184/344/459s 板面完整（无手或手仅在边缘）→ none
· 其余为手/刀/食材遮挡，板缘始终可辨 → mild
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'occlusion_phase0'
LABELS = OUT / 'labels_blind.json'
SOURCE = 'BV1j441157px'
EVIDENCE = 'experiments/occlusion_phase0/grid_BV1j441157px.jpg'

NONE = {5: '刀与碎屑，手在画面最左缘', 6: '空板与水滴', 7: '刀在板上，无手',
        14: '切好的青菜堆，无手', 19: '刀与碎屑，板面完整'}
MILD = {
    0: '双手持红椒在板上方',
    1: '手与刀切红椒圈',
    2: '手与刀，椒圈铺中部',
    3: '椒圈与右手，板缘完整',
    4: '椒圈与右手',
    8: '手按纸巾擦板，板缘可辨',
    9: '双手与青菜叶覆盖右侧',
    10: '手与刀切青菜，板缘完整',
    11: '双手处理青菜，板面部分被叶遮挡',
    12: '青菜段与手',
    13: '手与刀，大片叶子横跨',
    15: '手与青菜段',
    16: '手持红洋葱在板上方',
    17: '洋葱半 + 手 + 刀',
    18: '洋葱片 + 手 + 刀',
}


def main():
    data = json.loads(LABELS.read_text(encoding='utf-8'))
    manifest = json.loads((OUT / 'blind_v2_manifest.json').read_text(encoding='utf-8'))
    frames = [f for f in manifest['frames'] if f['source_bvid'] == SOURCE]
    counts = {}
    for slot, record in enumerate(frames):
        name = Path(record['file']).name
        if slot in NONE:
            entry = {'board_present': 'yes', 'occlusion_level': 'none',
                     'visible_box': None, 'full_box': None, 'note': NONE[slot]}
        else:
            entry = {'board_present': 'yes', 'occlusion_level': 'mild',
                     'visible_box': None, 'full_box': None,
                     'occluder': ['hand', 'knife', 'vegetable'],
                     'note': MILD.get(slot, '手/刀/食材遮挡，板缘可辨')}
        entry.update({'label_confidence': 'high', 'evidence': EVIDENCE,
                      'source_bvid': SOURCE, 'seconds': record['seconds']})
        data['labels'][name] = entry
        key = entry['occlusion_level']
        counts[key] = counts.get(key, 0) + 1
    data.setdefault('batch_log', []).append(
        {'source': SOURCE, 'frames': len(frames), 'counts': counts,
         'evidence': EVIDENCE, 'at': '2026-11-21T16:50:00+08:00'})
    data['counts'] = {'labelled_total': len(data['labels'])}
    LABELS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(SOURCE, counts, 'total', len(data['labels']))


if __name__ == '__main__':
    main()
