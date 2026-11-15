"""扫描本机未使用视频，找"白色圆板 / 圆砧墩"场景，用于补多来源正例。

每个视频取 3 帧（20%/50%/80%），只做人工看图，不跑模型。
"""
import json
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
INBOX = BASE.parent / '待导入视频'
OUT = BASE / 'experiments' / 'scan_white_round_20261115'
USED_IDS = set()
for mf in (BASE / 'experiments').glob('new_source_video_v*/frames_manifest.json'):
    data = json.loads(mf.read_text(encoding='utf-8'))
    for frame in data.get('frames', []):
        USED_IDS.add(Path(frame['file']).name.split('_')[0])


def main():
    (OUT / 'frames').mkdir(parents=True, exist_ok=True)
    records = []
    for video in sorted(INBOX.glob('*.mp4')):
        import re
        match = re.search(r'(BV\w+)', video.name)
        if not match:
            continue
        bvid = match.group(1)
        if bvid in USED_IDS:
            continue
        capture = cv2.VideoCapture(str(video))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            capture.release()
            continue
        for ratio in (0.2, 0.5, 0.8):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(total * ratio))
            ok, frame = capture.read()
            if not ok:
                continue
            name = '%s_%02d.jpg' % (bvid, int(ratio * 100))
            path = OUT / 'frames' / name
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            records.append({'video': str(video), 'bvid': bvid, 'ratio': ratio, 'file': str(path)})
        capture.release()
    (OUT / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 12)
    per_sheet = 12
    for start in range(0, len(records), per_sheet):
        canvas = Image.new('RGB', (1200, 560), 'white')
        for slot, rec in enumerate(records[start:start + per_sheet]):
            im = Image.open(rec['file']).convert('RGB')
            im.thumbnail((300, 160))
            canvas.paste(im, ((slot % 4) * 300, (slot // 4) * 175))
            ImageDraw.Draw(canvas).text(((slot % 4) * 300 + 2, (slot // 4) * 175 + 162),
                                        '%s %d%%' % (rec['bvid'][-4:], rec['ratio'] * 100), fill='black', font=font)
        canvas.save(OUT / ('sheet_%d.jpg' % (start // per_sheet + 1)), quality=88)
    print(json.dumps({'videos': len({r['bvid'] for r in records}), 'frames': len(records),
                      'sheets': (len(records) + per_sheet - 1) // per_sheet}, ensure_ascii=False))


if __name__ == '__main__':
    main()
