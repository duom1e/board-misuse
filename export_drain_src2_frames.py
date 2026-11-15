"""导出第二来源的候选帧（按命中时间均匀抽取），供人工核对。"""
import json
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
D = BASE / 'experiments' / 'drain_negative_src2_20261115'
VIDEO = next((BASE.parent / '待导入视频').glob('*BV18pRhBgEyw*.mp4'))
PICK = 16


def main():
    manifest = json.loads((D / 'manifest.json').read_text(encoding='utf-8'))
    cands = manifest['candidates']
    step = max(1, len(cands) // PICK)
    chosen = cands[::step][:PICK]
    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    saved = []
    for c in chosen:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(c['seconds'] * fps))
        ok, frame = capture.read()
        if not ok:
            continue
        path = D / 'images' / ('cand_%04ds.jpg' % c['seconds'])
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved.append({'file': str(path), 'seconds': c['seconds'], 'score': c['score']})
    capture.release()
    (D / 'review_list.json').write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 13)
    cols, rows = 4, 4
    canvas = Image.new('RGB', (cols * 300, rows * 190), 'white')
    for i, item in enumerate(saved[:cols * rows]):
        im = Image.open(item['file']).convert('RGB')
        im.thumbnail((300, 170))
        canvas.paste(im, ((i % cols) * 300, (i // cols) * 190))
        ImageDraw.Draw(canvas).text(((i % cols) * 300 + 3, (i // cols) * 190 + 172),
                                    '%ds %.2f' % (item['seconds'], item['score']), fill='black', font=font)
    canvas.save(D / 'review_sheet.jpg', quality=90)
    print(json.dumps({'saved': len(saved)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
