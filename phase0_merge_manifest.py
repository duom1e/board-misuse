"""阶段 0-d：把三批抽帧合并成盲测集 v2 的单一清单，并统计构成与缺口。"""
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'occlusion_phase0'
PLANS = ['sampling_plan.json', 'supplement_plan.json', 'supplement2_plan.json']
# 人工判定：来源是否有板（依据已查看的联系表）
SOURCE_KIND = {
    'BV1684y1r7HX': 'board',    # 沉浸式切菜，固定俯拍
    'BV1j441157px': 'board',    # 德叔助眠切菜
    'BV1wS4y1T78o': 'board',    # 顺德鱼生
    'BV1dh41187N8': 'board',    # 中式刀工
    'BV1ux411g7tW': 'board',    # 餐厅主厨刀工
    'BV1nm4y1Z7qs': 'board',    # 大厨切土豆丝
    'BV19b411v7Wz': 'noboard',  # 酒店后厨探秘（炒锅/成品/采访）
    'BV1ce4y1o74e': 'noboard',  # 净菜流水线
    'BV1Gz4y1y7UH': 'noboard',  # 大型酒店后厨
    'BV1Vc411b7ov': 'noboard',  # 米其林后厨：多数无板，仅 48s/93s 有白色砧板（逐帧判）
    'BV1krWG6KEUV': 'discard',  # 全程口播
}


def main():
    frames = []
    for plan in PLANS:
        path = OUT / plan
        if path.exists():
            frames += json.loads(path.read_text(encoding='utf-8'))['frames']
    by_source = defaultdict(list)
    for record in frames:
        by_source[record['source_bvid']].append(record)

    table = []
    for bvid, items in sorted(by_source.items()):
        table.append({'bvid': bvid, 'kind': SOURCE_KIND.get(bvid, 'unknown'),
                      'title': items[0].get('video_title', ''), 'frames': len(items)})

    board = sum(t['frames'] for t in table if t['kind'] == 'board')
    noboard = sum(t['frames'] for t in table if t['kind'] == 'noboard')
    discard = sum(t['frames'] for t in table if t['kind'] == 'discard')

    manifest = {
        'plan': 'occlusion_blind_v2',
        'status': 'frames_frozen_labels_pending',
        'rule': '按视频整段留出；每段 20 帧；抽样时刻在看模型前固定',
        'counts': {'board_frames': board, 'noboard_frames': noboard, 'discarded_frames': discard,
                   'total_kept': board + noboard, 'sources_board': sum(1 for t in table if t['kind'] == 'board'),
                   'sources_noboard': sum(1 for t in table if t['kind'] == 'noboard')},
        'gaps_vs_target': {
            'target_board': 120, 'have_board': board,
            'target_noboard': 200, 'have_noboard': noboard,
            'target_sources': 12, 'have_sources': sum(1 for t in table if t['kind'] in {'board', 'noboard'}),
        },
        'caveat': '同一来源内的帧高度相关，不能当作独立样本；指标需按来源分组报告',
        'sources': table,
        'frames': frames,
    }
    (OUT / 'blind_v2_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest['counts'], ensure_ascii=False))
    print(json.dumps(manifest['gaps_vs_target'], ensure_ascii=False))
    for t in table:
        print('%-14s %-8s %3d  %s' % (t['bvid'], t['kind'], t['frames'], t['title'][:34]))


if __name__ == '__main__':
    main()
