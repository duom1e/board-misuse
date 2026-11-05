"""更正分块探针：BV1dHckzwEBA_11 放大后是**炒锅**（有手柄、装切好的葱段），不是圆板。

因此：
  · 它在整帧被判"无板"是正确的，不是"正例漏检"；
  · 分块把它翻成"有板"属于**误报**，不是"找回"；
  · 探针判据因此不成立（负例出现了 1 次误报）。

保留原始结果，另写 results_corrected.json 与结论。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'vlm_tiling_probe_v1'
EVIDENCE = 'experiments/tiling_smoke_20261104/missed_11_zoom.jpg'


def main():
    data = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    for row in data['rows']:
        if row['image'] == 'BV1dHckzwEBA_11.jpg':
            row['reclassified'] = 'wok_not_board'
            row['reclassified_reason'] = ('放大可见碗状锅体与手柄、内有切好的葱段，是炒锅不是砧板；'
                                          '整帧判无板正确，分块判有板为误报')
            row['reclassified_evidence'] = EVIDENCE
    rows = data['rows']
    positives = [r for r in rows if r['kind'] == 'positive' and r.get('reclassified') != 'wok_not_board']
    woks = [r for r in rows if r.get('reclassified') == 'wok_not_board']
    negatives = [r for r in rows if r['kind'] == 'negative']
    corrected = {
        'true_positives': len(positives),
        'true_positives_full_frame': sum(1 for r in positives if r['full_frame_state'] == 'candidate_present'),
        'true_positives_recovered_by_tiling': sum(1 for r in positives
                                                  if r['full_frame_state'] != 'candidate_present'
                                                  and r['tiling_found_board']),
        'mislabeled_as_positive_actually_wok': len(woks),
        'tiling_false_alarms_on_wok': sum(1 for r in woks if r['tiling_found_board']),
        'negatives': len(negatives),
        'tiling_false_alarms_on_negatives': sum(1 for r in negatives if r['tiling_found_board']),
        'rule_met': False,
        'verdict': ('分块补救不成立：真正的圆板整帧就能检出（2/2），分块只把炒锅误判成板；'
                    '默认保持关闭（--tiling-fallback off）'),
        'also_corrects': 'experiments/full_fusion_review_20261104 把该图当作"正例漏检"，实为炒锅，应降级为"非板样本"',
    }
    (OUT / 'results_corrected.json').write_text(
        json.dumps({'corrected': corrected, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(corrected, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
