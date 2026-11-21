"""阶段 0-b3：双清单检索——补"有板素材"与"无板厨房素材"。

输出 experiments/occlusion_phase0/candidates_b3.json
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

BOARD_KEYWORDS = ['砧板 切菜 俯拍', '切配 台面 特写', '后厨 切菜 近景 无解说', '菜刀 砧板 切丝 教学']
NOBOARD_KEYWORDS = ['食堂 打饭 窗口', '后厨 炒锅 出餐', '厨房 洗碗 流水', '餐厅 传菜 前台']
BOARD_OK = ['切菜', '刀工', '切配', '砧板', '案板', '备菜']
NOBOARD_OK = ['炒锅', '打饭', '出餐', '洗碗', '传菜', '厨房', '后厨', '食堂']
BAD = ['游戏', '动画', '测评', '开箱', '装修', '推荐', '神器', '避坑', '探店', '直播', '教程', '广告']


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


def search(keywords, accept, used):
    hits = {}
    for keyword in keywords:
        try:
            rows = (requests.get('https://api.bilibili.com/x/web-interface/search/type',
                                 params={'search_type': 'video', 'keyword': keyword},
                                 headers={'User-Agent': UA, 'Referer': 'https://www.bilibili.com/'},
                                 timeout=25).json().get('data') or {}).get('result') or []
        except Exception as exc:
            print('failed', keyword, exc)
            continue
        for row in rows:
            bvid = row.get('bvid')
            title = re.sub('<[^>]+>', '', row.get('title', ''))
            if not bvid or bvid in used or bvid in hits:
                continue
            if not any(word in title for word in accept):
                continue
            if any(word in title for word in BAD):
                continue
            hits[bvid] = {'title': title, 'duration': row.get('duration'), 'play': row.get('play'),
                          'keyword': keyword}
    return hits


def main():
    used = seen()
    board = search(BOARD_KEYWORDS, BOARD_OK, used)
    used |= set(board)
    noboard = search(NOBOARD_KEYWORDS, NOBOARD_OK, used)
    payload = {'board_candidates': board, 'noboard_candidates': noboard}
    (OUT / 'candidates_b3.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('board candidates', len(board))
    for bvid, info in sorted(board.items(), key=lambda kv: -(kv[1]['play'] or 0))[:10]:
        print('  B', bvid, info['duration'], info['play'], info['title'][:40])
    print('noboard candidates', len(noboard))
    for bvid, info in sorted(noboard.items(), key=lambda kv: -(kv[1]['play'] or 0))[:10]:
        print('  N', bvid, info['duration'], info['play'], info['title'][:40])


if __name__ == '__main__':
    main()
