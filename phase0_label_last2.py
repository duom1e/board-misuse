"""把最后两段有板来源（BV1wS4y1T78o、BV1dh41187N8）的标注写入 labels_blind.json。

依据：grid_BV1wS4y1T78o.jpg、grid_BV1dh41187N8.jpg。
· 鱼生段 78/83s 是装盆画面，无板 → no；41s 为模糊特写，无法确认板 → unclear
· 刀工段 433s 为手持成品特写，板基本出画 → unclear
· 其余均为手/刀/食材遮挡但板缘可辨 → mild
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'occlusion_phase0'
LABELS = OUT / 'labels_blind.json'

PLAN = {
    'BV1wS4y1T78o': {
        'evidence': 'experiments/occlusion_phase0/grid_BV1wS4y1T78o.jpg',
        'no': {18: '不锈钢盆中的鱼丝，画面内无板', 19: '盆中鱼丝，无板'},
        'unclear': {9: '模糊特写（白色鱼柳），无法确认板是否在画面内'},
        'mild_default': '手/刀/鱼柳遮挡，圆墩边缘可辨',
    },
    'BV1dh41187N8': {
        'evidence': 'experiments/occlusion_phase0/grid_BV1dh41187N8.jpg',
        'no': {},
        'unclear': {18: '双手持成品银针丝特写，白板基本出画'},
        'mild_default': '手/刀/萝卜条遮挡，白色圆板边缘可见',
    },
}


def main():
    data = json.loads(LABELS.read_text(encoding='utf-8'))
    manifest = json.loads((OUT / 'blind_v2_manifest.json').read_text(encoding='utf-8'))
    for source, spec in PLAN.items():
        frames = [f for f in manifest['frames'] if f['source_bvid'] == source]
        counts = {}
        for slot, record in enumerate(frames):
            name = Path(record['file']).name
            if slot in spec['no']:
                entry = {'board_present': 'no', 'occlusion_level': 'not_applicable',
                         'visible_box': None, 'full_box': None, 'note': spec['no'][slot]}
            elif slot in spec['unclear']:
                entry = {'board_present': 'unclear', 'occlusion_level': 'not_applicable',
                         'visible_box': None, 'full_box': None, 'note': spec['unclear'][slot]}
            else:
                entry = {'board_present': 'yes', 'occlusion_level': 'mild',
                         'visible_box': None, 'full_box': None,
                         'occluder': ['hand', 'knife', 'vegetable'],
                         'note': spec['mild_default']}
            entry.update({'label_confidence': 'high' if entry['board_present'] != 'unclear' else 'low',
                          'evidence': spec['evidence'], 'source_bvid': source,
                          'seconds': record['seconds']})
            data['labels'][name] = entry
            key = 'no' if entry['board_present'] == 'no' else (
                'unclear' if entry['board_present'] == 'unclear' else entry['occlusion_level'])
            counts[key] = counts.get(key, 0) + 1
        data.setdefault('batch_log', []).append(
            {'source': source, 'frames': len(frames), 'counts': counts,
             'evidence': spec['evidence'], 'at': '2026-11-21T17:10:00+08:00'})
        print(source, counts)
    data['counts'] = {'labelled_total': len(data['labels'])}
    LABELS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    from collections import Counter
    tally = Counter((v['board_present'], v['occlusion_level']) for v in data['labels'].values())
    print('total', len(data['labels']), dict(tally))


if __name__ == '__main__':
    main()
