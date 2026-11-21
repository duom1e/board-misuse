"""检修脚本：校验本轮阶段 0 产物的完整性与一致性（只读检查 + 报告，不改数据）。"""
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'occlusion_phase0'
problems = []
notes = []


def check(condition, message):
    if not condition:
        problems.append(message)


def main():
    labels_path = OUT / 'labels_blind.json'
    manifest_path = OUT / 'blind_v2_manifest.json'
    check(labels_path.exists(), 'labels_blind.json 缺失')
    check(manifest_path.exists(), 'blind_v2_manifest.json 缺失')
    labels = json.loads(labels_path.read_text(encoding='utf-8'))['labels']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    # 1) 标签里的帧是否都真实存在
    missing = [name for name in labels if not (OUT / 'frames' / name).exists()]
    check(not missing, '有 %d 个标签指向不存在的帧: %s' % (len(missing), missing[:5]))

    # 2) 帧是否在 manifest 里登记
    manifest_names = {Path(f['file']).name for f in manifest['frames']}
    unregistered = [name for name in labels if name not in manifest_names]
    check(not unregistered, '有 %d 个标签帧未在 manifest 登记: %s' % (len(unregistered), unregistered[:5]))

    # 3) 字段完整性
    required = {'board_present', 'occlusion_level', 'visible_box', 'full_box'}
    for name, entry in labels.items():
        lack = required - set(entry)
        if lack:
            problems.append('%s 缺字段 %s' % (name, sorted(lack)))
            break

    # 4) board_present 与 occlusion_level 的取值一致性
    for name, entry in labels.items():
        if entry['board_present'] == 'yes' and entry['occlusion_level'] not in {'none', 'mild', 'severe'}:
            problems.append('%s: yes 但 occlusion_level=%s' % (name, entry['occlusion_level']))
            break
        if entry['board_present'] != 'yes' and entry['occlusion_level'] != 'not_applicable':
            problems.append('%s: %s 但 occlusion_level=%s'
                            % (name, entry['board_present'], entry['occlusion_level']))
            break

    # 5) 重复标签（同名不同内容）
    tally = Counter((e['board_present'], e['occlusion_level']) for e in labels.values())
    notes.append('标签分布：' + json.dumps({str(k): v for k, v in tally.items()}, ensure_ascii=False))

    # 6) 磁盘上的帧数 vs manifest 帧数
    on_disk = {p.name for p in (OUT / 'frames').glob('*.jpg')}
    check(len(on_disk) >= len(manifest_names),
          '磁盘帧数 %d < manifest 帧数 %d' % (len(on_disk), len(manifest_names)))
    extra = on_disk - manifest_names
    window_frames = [name for name in extra if name.startswith('w_')]
    # 其余未登记帧应来自"筛除但保留"的补采批次（severe_plan*.json）
    plan_frames = set()
    for plan_name in ('severe_plan.json', 'severe_plan2.json'):
        plan_path = OUT / plan_name
        if plan_path.exists():
            payload = json.loads(plan_path.read_text(encoding='utf-8'))
            plan_frames |= {Path(f['file']).name for f in payload.get('frames', [])}
    screened = [name for name in extra if name in plan_frames]
    unexpected = [name for name in extra if not name.startswith('w_') and name not in plan_frames]
    notes.append('磁盘帧 %d、manifest 登记 %d、窗口帧 %d、筛除批次帧 %d、意外 %d'
                 % (len(on_disk), len(manifest_names), len(window_frames), len(screened), len(unexpected)))
    check(not unexpected, '有 %d 个帧无法追溯来源: %s' % (len(unexpected), unexpected[:5]))

    # 7) 密集窗口帧的完整性
    windows = ['BV1wS4y1T78o', 'BV1dh41187N8', 'BV1ux411g7tW']
    for bvid in windows:
        count = len(list((OUT / 'frames').glob('w_%s_*.jpg' % bvid)))
        notes.append('窗口帧 %s = %d' % (bvid, count))
        check(count > 0, '窗口 %s 没有帧' % bvid)

    # 8) 评估结果与标签是否对得上
    baseline = OUT / 'baseline_A_eval.json'
    if baseline.exists():
        report = json.loads(baseline.read_text(encoding='utf-8'))['report']
        total = report['frames']
        check(total == len(labels), 'baseline 评估帧数 %d 与标签数 %d 不一致' % (total, len(labels)))
        notes.append('baseline 评估帧数 = %d' % total)

    # 9) 关键脚本是否存在且可解析
    scripts = ['phase0_dense_windows.py', 'eval_persistence.py', 'eval_occlusion_suite.py',
               'phase0_merge_manifest.py', 'phase0_make_grids.py']
    import ast
    for script in scripts:
        path = BASE / script
        check(path.exists(), '脚本缺失: %s' % script)
        if path.exists():
            try:
                ast.parse(path.read_text(encoding='utf-8'))
            except SyntaxError as exc:
                problems.append('%s 语法错误: %s' % (script, exc))

    print('== 问题 ==')
    print('无' if not problems else '\n'.join('- ' + p for p in problems))
    print('== 说明 ==')
    print('\n'.join('- ' + n for n in notes))


if __name__ == '__main__':
    main()
