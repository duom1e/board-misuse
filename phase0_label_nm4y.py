"""把 BV1nm4y1Z7qs 的 20 帧遮挡标注写入 labels_blind.json。

依据：grid_BV1nm4y1Z7qs.jpg。前 7 帧（5–30s）画面是托盘/人物，板不在画面内 → no。
从 35s 起为绿色圆板，手/刀/土豆丝遮挡，板缘始终可见 → mild。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'occlusion_phase0'
LABELS = OUT / 'labels_blind.json'
SOURCE = 'BV1nm4y1Z7qs'
EVIDENCE = 'experiments/occlusion_phase0/grid_BV1nm4y1Z7qs.jpg'

NO_BOARD = {0: '托盘与人物，板不在画面', 1: '托盘', 2: '托盘', 3: '托盘', 4: '托盘',
            5: '托盘与人物手臂', 6: '人物头部特写，无板'}
MILD = {
    7: '绿板 + 刀 + 土豆，手在右',
    8: '手持刀压在板上',
    9: '双手与刀，板缘可见',
    10: '手与刀切片，板面大可见',
    11: '双手切片，板缘完整',
    12: '双手与刀，遮挡居中',
    13: '手与刀，板缘完整',
    14: '手掌推土豆丝，板缘可见',
    15: '手与刀，丝铺在中部',
    16: '丝与刀遮中部，圆板外缘可见',
    17: '土豆丝铺开，圆板外缘仍可辨',
    18: '丝铺满中部，手在两侧',
    19: '手与刀，丝在右下',
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
        else:
            entry = {'board_present': 'yes', 'occlusion_level': 'mild',
                     'visible_box': None, 'full_box': None,
                     'occluder': ['hand', 'knife', 'vegetable'], 'note': MILD[slot]}
        entry.update({'label_confidence': 'high', 'evidence': EVIDENCE,
                      'source_bvid': SOURCE, 'seconds': record['seconds']})
        data['labels'][name] = entry
        key = entry['board_present'] if entry['board_present'] == 'no' else entry['occlusion_level']
        counts[key] = counts.get(key, 0) + 1
    data.setdefault('batch_log', []).append(
        {'source': SOURCE, 'frames': len(frames), 'counts': counts,
         'evidence': EVIDENCE, 'at': '2026-11-21T16:20:00+08:00'})
    data['counts'] = {'labelled_total': len(data['labels'])}
    LABELS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(SOURCE, counts, 'total', len(data['labels']))


if __name__ == '__main__':
    main()
