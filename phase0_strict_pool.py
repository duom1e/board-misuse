"""阶段 0-a（严格版）：统计"在本项目任何文件里出现过的 BV 号"，得到真正未被使用过的视频。

上一版只排除了当前训练清单与少数开发锚点，但历史上 v1–v8 批次、board_seg、pool 等
都消费过大量来源；只要 BV 号在 outputs 下任何 json/txt/md 里出现过，就一律视为"已接触"。
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
INBOX = BASE.parent / '待导入视频'
OUT = BASE / 'experiments' / 'occlusion_phase0'
BV = re.compile(r'BV[0-9A-Za-z]{10}')
SKIP_DIRS = {'github_export_20261025', 'github_export_20261026_latest',
             'github_export_20261029', 'github_export_20261029_final'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seen = {}
    for path in BASE.rglob('*'):
        if path.is_dir() or path.suffix.lower() not in {'.json', '.txt', '.md', '.py'}:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        for match in BV.finditer(text):
            seen.setdefault(match.group(0), set()).add(str(path.relative_to(BASE)))

    videos = []
    for path in sorted(INBOX.glob('*.mp4')):
        match = BV.search(path.name)
        if match:
            videos.append({'bvid': match.group(0), 'file': str(path),
                           'mb': round(path.stat().st_size / 1048576, 1)})

    clean = [v for v in videos if v['bvid'] not in seen]
    payload = {
        'rule': 'BV 号在 outputs 下任何 json/txt/md/py 出现过即视为已接触（排除 github_export 快照）',
        'ever_seen_count': len(seen),
        'total_local_videos': len(videos),
        'clean': clean,
        'seen_examples': {k: sorted(v)[:3] for k, v in list(seen.items())[:15]},
    }
    (OUT / 'source_pool_strict.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ever seen bvids', len(seen))
    print('local videos', len(videos))
    print('CLEAN', len(clean))
    for item in clean:
        print(' ', item['bvid'], item['mb'], 'MB', Path(item['file']).name[:60])


if __name__ == '__main__':
    main()
