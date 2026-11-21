"""阶段 0-f（P0）：专门为 severe 档找素材。

动机：固定机位切菜视频产生不了 severe（板始终完整在画面内）。
所以改找"手持 / 第一视角 / 手机实拍"的后厨画面——板更容易被手、人、容器挡住大半或只剩一条边。

流程：检索 → 挑 3 段下载 → 每段抽 25 帧 → 出 5×5 大图供人工挑 severe。
不跑模型、不训练。
"""
import json
import re
import subprocess
from pathlib import Path

import cv2
import requests

BASE = Path(__file__).resolve().parent
INBOX = BASE.parent / '待导入视频'
OUT = BASE / 'experiments' / 'occlusion_phase0'
PY = BASE.parent / 'work' / 'board-env' / 'Scripts' / 'python.exe'
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')
BV = re.compile(r'BV[0-9A-Za-z]{10}')
KEYWORDS = ['第一视角 后厨 上班', '后厨 vlog 手机实拍 切菜', '厨师 视角 切配 特写', '后厨 真实 手持 拍摄']
PREFER = ['第一视角', 'vlog', '实拍', '手持', '上班', '记录']
BAD = ['游戏', '动画', '测评', '开箱', '装修', '推荐', '神器', '避坑', '探店', '直播', '广告', '教程']


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
    candidates = {}
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
            if not bvid or bvid in used or bvid in candidates:
                continue
            if not any(word in title for word in ['后厨', '厨房', '切菜', '切配', '备菜']):
                continue
            if any(word in title for word in BAD):
                continue
            score = sum(1 for word in PREFER if word in title)
            candidates[bvid] = {'title': title, 'duration': row.get('duration'),
                                'play': row.get('play'), 'score': score}
    ordered = sorted(candidates.items(), key=lambda kv: (-kv[1]['score'], -(kv[1]['play'] or 0)))
    picks = ordered[:3]
    print('picked', [(b, v['title'][:30]) for b, v in picks])

    records = []
    for bvid, info in picks:
        target = INBOX
        subprocess.run([str(PY), '-m', 'yt_dlp', '--no-warnings', '-f', '30032/30016/30064/best',
                        '-o', str(target / '%(title).40s [%(id)s].%(ext)s'),
                        f'https://www.bilibili.com/video/{bvid}'], capture_output=True)
        video = next(target.glob(f'*{bvid}*.mp4'), None)
        if video is None:
            print('download failed', bvid)
            continue
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps
        for k in range(25):
            ratio = 0.05 + 0.90 * k / 24
            seconds = round(duration * ratio, 2)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
            ok, frame = capture.read()
            if not ok:
                continue
            name = '%s_%06.1fs.jpg' % (bvid, seconds)
            path = OUT / 'frames' / name
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            records.append({'source_bvid': bvid, 'video_title': info['title'], 'seconds': seconds,
                            'file': str(path)})
        capture.release()

    (OUT / 'severe_plan.json').write_text(json.dumps({
        'plan': 'occlusion_severe_v1',
        'why': '固定机位无法产生 severe，改找手持/第一视角',
        'sources': [{'bvid': b, 'title': v['title']} for b, v in picks],
        'frames': records,
        'model_inference_performed': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 12)
    for bvid, _ in picks:
        frames = [r for r in records if r['source_bvid'] == bvid]
        if not frames:
            continue
        canvas = Image.new('RGB', (300 * 5, 188 * 5), 'white')
        for slot, record in enumerate(frames[:25]):
            image = Image.open(record['file']).convert('RGB').resize((300, 170), Image.LANCZOS)
            x, y = (slot % 5) * 300, (slot // 5) * 188
            canvas.paste(image, (x, y))
            ImageDraw.Draw(canvas).text((x + 2, y + 171), '#%d %.0fs' % (slot, record['seconds']),
                                        fill='black', font=font)
        canvas.save(OUT / ('severe_grid_%s.jpg' % bvid), quality=90)
        print('grid', bvid, len(frames))


if __name__ == '__main__':
    main()
