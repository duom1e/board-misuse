"""阶段 0-a：挑出"从未参与训练/开发/盲测"的视频来源，作为遮挡盲测集 v2 的候选池。

判定依据（全部读自项目已有文件，不靠记忆）：
  · 训练来源：multisource_paired15 的 mixed.txt、pool4 的 train.txt，以及各额外正负例目录里的图片名
  · 开发来源：白板锚点、切鸡锚点、食堂巡检段、wok 线、U7Zh 中景
  · 盲测 v1：两段已用于 2026-11-15 盲测的视频
输出：experiments/occlusion_phase0/source_pool.json
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent          # outputs/
INBOX = BASE.parent / '待导入视频'
OUT = BASE / 'experiments' / 'occlusion_phase0'

TRAIN_LISTS = [
    'experiments/multisource_paired15_20261111/mixed.txt',
    'experiments/pool4_positive15_seed73_20261115/train.txt',
]
TRAIN_IMAGE_DIRS = [
    'experiments/kitchen_neg2_20261115/images',
    'experiments/newpos_20261115/images',
    'experiments/wood_positive_20261115/images',
    'experiments/round_block_positive_20261115/images',
    'experiments/scan_white_round_20261115/frames',
    'experiments/drain_negative_src2_20261115/images',
    'experiments/blind_test_v1/frames',
]
DEV_AND_BLIND = {
    'BV1Go4y1o7ym',  # 白色圆板锚点
    'BV1264y1F7a6',  # 切鸡圆墩锚点
    'BV1bY4y1Y7v5',  # 食堂巡检 880–920s（留出段）
    'BV1b1ixB5ERA',  # 炒锅线
    'BV1pu411U7Zh',  # 中景后厨（val 组）
    'BV1Mg411U7SV',  # 盲测 v1
    'BV1Zq4y1j7mm',  # 盲测 v1
}
BV = re.compile(r'BV\w+')


def collect_training_bvids():
    found = set()
    for rel in TRAIN_LISTS:
        path = BASE / rel
        if path.exists():
            for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
                match = BV.search(line)
                if match:
                    found.add(match.group(0))
    for rel in TRAIN_IMAGE_DIRS:
        directory = BASE / rel
        if not directory.exists():
            continue
        for item in directory.glob('*.jpg'):
            match = BV.search(item.name)
            if match:
                found.add(match.group(0))
    return found


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    training = collect_training_bvids()
    videos = []
    for path in sorted(INBOX.glob('*.mp4')):
        match = BV.search(path.name)
        if not match:
            continue
        videos.append({'bvid': match.group(0), 'file': str(path),
                       'mb': round(path.stat().st_size / 1048576, 1)})
    used = training | DEV_AND_BLIND
    clean = [v for v in videos if v['bvid'] not in used]
    payload = {
        'note': '候选池 = 本机 52 段视频中，排除训练/开发/盲测 v1 的全部来源',
        'trained_or_dev_bvids': sorted(used),
        'total_videos': len(videos),
        'clean_candidates': clean,
    }
    (OUT / 'source_pool.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('total', len(videos), 'used', len(used), 'clean', len(clean))
    for item in clean:
        print(item['bvid'], item['mb'], 'MB', Path(item['file']).name[:50])


if __name__ == '__main__':
    main()
