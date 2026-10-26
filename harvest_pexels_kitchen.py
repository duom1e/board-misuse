"""Harvest kitchen photos from Pexels via its public photo sitemaps.

Pexels licence: photos are free to use, attribution not required (still recorded here).
Sitemaps give 50k photo-page URLs each with descriptive slugs, so kitchen scenes can be
found by slug filtering without touching the (bot-protected) search pages.
"""
import gzip
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
OUT = BASE / 'pexels_kitchen'
INDEX = 'https://www.pexels.com/sitemaps/en-US/photo-sitemap.xml.gz'
KEYWORDS = ('kitchen', 'chef', 'restaurant', 'canteen', 'cook', 'catering', 'gastronomy')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36',
           'Accept': 'application/xml,text/xml,*/*'}
PATTERN = re.compile(r'<loc>(https://www\.pexels\.com/photo/([^<]+)-(\d+)/)</loc>')


def fetch(url, tries=3):
    for attempt in range(tries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            return response.content
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def sitemaps():
    text = gzip.decompress(fetch(INDEX)).decode('utf-8')
    return re.findall(r'<loc>([^<]+)</loc>', text)


def scan(url):
    hits = []
    text = gzip.decompress(fetch(url)).decode('utf-8', 'ignore')
    for match in PATTERN.finditer(text):
        page, slug, photo_id = match.groups()
        slug_lower = slug.lower()
        if any(word in slug_lower for word in KEYWORDS):
            hits.append({'id': int(photo_id), 'slug': slug, 'page': page})
    return hits


def main():
    OUT.mkdir(exist_ok=True)
    maps = sitemaps()
    print('sitemaps', len(maps), flush=True)
    found = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(scan, url): url for url in maps}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                for hit in future.result():
                    found[hit['id']] = hit
            except Exception as exc:  # noqa: BLE001
                print('failed', futures[future], exc, flush=True)
            if count % 10 == 0:
                print(f'{count}/{len(maps)} sitemaps, {len(found)} candidates', flush=True)
    (OUT / 'candidates.json').write_text(json.dumps({
        'source': 'Pexels public photo sitemaps (en-US)',
        'license': 'Pexels licence: free to use, attribution not required',
        'keywords': KEYWORDS,
        'candidates': sorted(found.values(), key=lambda row: row['id']),
    }, indent=2, ensure_ascii=False))
    print('TOTAL', len(found), flush=True)


if __name__ == '__main__':
    main()
