"""把 BV1684y1r7HX 的 20 帧遮挡标注写入 labels_blind.json。

判定依据：grid_BV1684y1r7HX.jpg（5×4 放大图，每格 300×170），按《遮挡分级标注规范_v1》
"能否沿可见边推出四个角"分档。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
LABELS = BASE / 'experiments' / 'occlusion_phase0' / 'labels_blind.json'
SOURCE = 'BV1684y1r7HX'
EVIDENCE = 'experiments/occlusion_phase0/grid_BV1684y1r7HX.jpg'

# 逐帧：槽位 -> (档位, 遮挡物, 备注)
FRAMES = {
    0: ('mild', ['hand'], '左手在板上缘，板右/下/左可辨'),
    1: ('mild', ['hand', 'knife', 'vegetable'], '双手持刀切洋葱，遮挡居中'),
    2: ('mild', ['hand', 'knife', 'meat'], '刀与肉在板中部，四边可见'),
    3: ('mild', ['hand', 'meat'], '切片铺在板上，手在右上'),
    4: ('mild', ['hand', 'knife', 'vegetable'], '黄瓜段遮右下，板面仍成形'),
    5: ('none', [], '空板，四边完整'),
    6: ('mild', ['hand', 'knife', 'vegetable'], '手与刀在板中，边线完整'),
    7: ('mild', ['hand', 'knife', 'vegetable'], '同上，土豆片少量遮挡'),
    8: ('none', ['knife'], '仅刀落在板上，板面完整'),
    9: ('mild', ['hand', 'knife', 'vegetable'], '手压土豆，板边仍可推'),
    10: ('mild', ['hand', 'knife', 'vegetable'], '土豆丝铺开，板缘可辨'),
    11: ('mild', ['hand', 'knife', 'vegetable'], '同上'),
    12: ('mild', ['hand', 'knife'], '手掌压刀，遮板中部'),
    13: ('mild', ['hand', 'knife', 'vegetable'], '手与刀在板上'),
    14: ('mild', ['hand', 'vegetable'], '手与土豆片遮挡右侧'),
    15: ('mild', ['hand', 'knife', 'vegetable'], '长条土豆丝横跨板面'),
    16: ('none', ['knife'], '仅菜刀，板面完整'),
    17: ('none', ['knife'], '菜刀与水渍，板面完整'),
    18: ('mild', ['hand', 'knife', 'vegetable'], '长葱叶遮右侧，左/下可辨'),
    19: ('mild', ['hand', 'vegetable'], '葱花堆在右下，板面大部分可见'),
}


def main():
    data = json.loads(LABELS.read_text(encoding='utf-8')) if LABELS.exists() else {
        'plan': 'occlusion_blind_v2', 'labelled_by': 'assistant visual screening（单标注者，非专家）',
        'single_rater': True, 'rule': '《遮挡分级标注规范_v1》', 'labels': {}}
    counts = {}
    # 通过 manifest 精确定位文件名（避免秒数格式差异）
    manifest = json.loads((BASE / 'experiments' / 'occlusion_phase0' / 'blind_v2_manifest.json')
                          .read_text(encoding='utf-8'))
    frames = [f for f in manifest['frames'] if f['source_bvid'] == SOURCE]
    for slot, (level, occluders, note) in FRAMES.items():
        record = frames[slot]
        name = Path(record['file']).name
        data['labels'][name] = {
            'board_present': 'yes',
            'occlusion_level': level,
            'visible_box': None,
            'full_box': None,
            'occluder': occluders,
            'note': note,
            'label_confidence': 'high',
            'evidence': EVIDENCE,
            'source_bvid': SOURCE,
            'seconds': record['seconds'],
        }
        counts[level] = counts.get(level, 0) + 1
    data['batch_log'] = data.get('batch_log', [])
    data['batch_log'].append({'source': SOURCE, 'frames': len(FRAMES), 'counts': counts,
                              'evidence': EVIDENCE, 'at': '2026-11-21T16:05:00+08:00'})
    data['counts'] = {'labelled_total': len(data['labels'])}
    LABELS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(SOURCE, counts, 'total labels', len(data['labels']))


if __name__ == '__main__':
    main()
