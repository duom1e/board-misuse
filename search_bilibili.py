"""在 B 站搜"后厨"类视频，筛出像真实后厨画面的候选（只看元数据，不下载）。"""
import json
import re
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / 'bilibili_search.json'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36'
KEYWORDS = ['后厨', '后厨 切菜', '食堂后厨', '餐厅后厨', '厨房 砧板 切肉']
BAD_WORDS = ['游戏', '动画', '鬼畜', '教程', 'asmr', '测评', '开箱', '装修', '厨师机', '模型', 'MAD', '解说']


def search(keyword, page=1):
    url = 'https://api.bilibili.com/x/web-interface/search/type'
    params = {'search_type': 'video', 'keyword': keyword, 'page': page}
    headers = {'User-Agent': UA, 'Referer': 'https://www.bilibili.com/'}
    response = requests.get(url, params=params, headers=headers, timeout=25)
    data = response.json()
    if data.get('code') != 0:
        return []
    return data['data'].get('result', []) or []


def clean(title):
    return re.sub(r'<[^>]+>', '', title)


def main():
    seen = {}
    for keyword in KEYWORDS:
        try:
            rows = search(keyword)
        except Exception as exc:  # noqa: BLE001
            print('search failed', keyword, exc)
            continue
        for row in rows:
            title = clean(row.get('title', ''))
            if any(word.lower() in title.lower() for word in BAD_WORDS):
                continue
            bvid = row.get('bvid')
            if not bvid or bvid in seen:
                continue
            seen[bvid] = {'bvid': bvid, 'title': title, 'author': row.get('author'),
                          'duration': row.get('duration'), 'play': row.get('play'),
                          'pubdate': row.get('pubdate'), 'keyword': keyword,
                          'url': f'https://www.bilibili.com/video/{bvid}'}
        print(keyword, '累计', len(seen), flush=True)
        time.sleep(1)
    rows = sorted(seen.values(), key=lambda r: -(r.get('play') or 0))
    OUT.write_text(json.dumps({'keywords': KEYWORDS, 'candidates': rows}, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print('候选', len(rows), '->', OUT)
    for row in rows[:15]:
        print('  ', row['bvid'], row['duration'], '播放', row['play'], row['title'][:40])


if __name__ == '__main__':
    main()
