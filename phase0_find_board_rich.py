"""阶段 0-b2：找"板面清楚、遮挡频繁"的新来源（沉浸式切菜类），补足盲测集。

上一批 6 段里只有 1 段（BV1684y1r7HX 沉浸式切菜）是板面素材，其余是炒锅/采访/流水线。
本脚本只检索候选，不下载。
"""
import json
import re
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'occlusion_phase0'
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')
BV = re.compile(r'BV[0-9A-Za-z]{10}')
KEYWORDS = ['沉浸式切菜 无解说', '切菜 asmr 砧板', '刀工 切配 特写 砧板', '后厨 切菜 第一视角']
BAD = ['游戏', '动画', '测评', '开箱', '装修', '推荐', '神器', '避坑', '探店', '直播', '教程']


def seen():
    found = set()
    for path in BASE.rglob('*'):
        if path.is_dir() or path.suffix.lower() not in {'.json', '.txt', '.md', '.py'}:
            continue
        try:
            found.update(m.group(0) for m in BV.finditer(path.read_text(encoding='utf-8', errors='ignore')))
        except Exception:
            continue
    return found


def main():
    used = seen()
    hits = {}
    for keyword in KEYWORDS:
        try:
            rows = (requests.get('https://api.bilibili.com/x/web-interface/search/type',
                                 params={'search_type': 'video', 'keyword': keyword},
                                 headers={'User-Agent': UA, 'Referer': 'https://www.bilibili.com/'},
                                 timeout=25).json().get('data') or {}).get('result') or []
        except Exception as exc:
            print('search failed', keyword, exc)
            continue
        for row in rows:
            bvid = row.get('bvid')
            title = re.sub('<[^>]+>', '', row.get('title', ''))
            if not bvid or bvid in used or bvid in hits:
                continue
            if not any(word in title for word in ['切菜', '刀工', '切配', '砧板', '案板', '备菜']):
                continue
            if any(word in title for word in BAD):
                continue
            hits[bvid] = {'title': title, 'duration': row.get('duration'), 'play': row.get('play')}
    ordered = sorted(hits.items(), key=lambda kv: -(kv[1]['play'] or 0))
    (OUT / 'board_rich_candidates.json').write_text(json.dumps(dict(ordered), ensure_ascii=False, indent=2),
                                                    encoding='utf-8')
    print('candidates', len(ordered))
    for bvid, info in ordered[:14]:
        print(bvid, info['duration'], info['play'], info['title'][:44])


if __name__ == '__main__':
    main()
