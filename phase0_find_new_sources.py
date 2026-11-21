"""阶段 0-b：为遮挡盲测集 v2 找**从未出现在本项目任何文件里**的新视频。

严格规则的结果是：本机 52 段视频全部已接触（CLEAN 0），所以盲测集必须用新下载素材。
本脚本只负责检索并打印候选，不下载。
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
KEYWORDS = [
    '中央厨房 切配 车间',
    '食堂 后厨 切菜 实拍',
    '餐厅 后厨 备菜 一天',
    '后厨 切配 工作 记录',
    '酒店 后厨 砧板 切菜',
]
BAD = ['游戏', '动画', '测评', '开箱', '装修', '教程', '推荐', '神器', '避坑', '探店', '广告', '直播回放']


def seen_bvids():
    seen = set()
    for path in BASE.rglob('*'):
        if path.is_dir() or path.suffix.lower() not in {'.json', '.txt', '.md', '.py'}:
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        seen.update(m.group(0) for m in BV.finditer(text))
    return seen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seen = seen_bvids()
    hits = {}
    for keyword in KEYWORDS:
        try:
            response = requests.get(
                'https://api.bilibili.com/x/web-interface/search/type',
                params={'search_type': 'video', 'keyword': keyword},
                headers={'User-Agent': UA, 'Referer': 'https://www.bilibili.com/'},
                timeout=25).json()
            rows = (response.get('data') or {}).get('result') or []
        except Exception as exc:
            print('search failed', keyword, exc)
            continue
        for row in rows:
            bvid = row.get('bvid')
            title = re.sub('<[^>]+>', '', row.get('title', ''))
            if not bvid or bvid in seen or bvid in hits:
                continue
            if not any(word in title for word in ['后厨', '厨房', '食堂', '切菜', '切配', '备菜', '砧板', '案板']):
                continue
            if any(word in title for word in BAD):
                continue
            hits[bvid] = {'title': title, 'duration': row.get('duration'),
                          'play': row.get('play'), 'keyword': keyword}
    ordered = sorted(hits.items(), key=lambda kv: -(kv[1]['play'] or 0))
    (OUT / 'new_source_candidates.json').write_text(
        json.dumps(dict(ordered), ensure_ascii=False, indent=2), encoding='utf-8')
    print('candidates', len(ordered))
    for bvid, info in ordered[:20]:
        print(bvid, info['duration'], info['play'], info['title'][:46])


if __name__ == '__main__':
    main()
