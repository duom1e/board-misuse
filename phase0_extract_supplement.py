"""阶段 0-c2：为补充的 3 段"板面素材"抽帧（时刻先固定），并生成联系表 7–9。

第一批 6 段里只有 1 段是板面素材，因此补充 3 段刀工/切菜教学视频。
"""
import cv2
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
INBOX = BASE.parent / '待导入视频'
OUT = BASE / 'experiments' / 'occlusion_phase0'
SOURCES = [
    ('BV1j441157px', '德叔助眠·切菜（8:04）'),
    ('BV1wS4y1T78o', '顺德鱼生刀工（1:28）'),
    ('BV1dh41187N8', '中式刀工演示·银针丝（8:01）'),
]
PER_VIDEO = 20


def main():
    (OUT / 'frames').mkdir(parents=True, exist_ok=True)
    records = []
    for bvid, title in SOURCES:
        video = next(INBOX.glob(f'*{bvid}*.mp4'))
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps
        for k in range(PER_VIDEO):
            ratio = 0.05 + 0.90 * k / (PER_VIDEO - 1)
            seconds = round(duration * ratio, 2)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
            ok, frame = capture.read()
            if not ok:
                continue
            name = '%s_%06.1fs.jpg' % (bvid, seconds)
            path = OUT / 'frames' / name
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            records.append({'source_bvid': bvid, 'video_title': title, 'seconds': seconds,
                            'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        capture.release()
        print(bvid, 'duration', round(duration, 1), 'frames',
              len([r for r in records if r['source_bvid'] == bvid]), flush=True)

    (OUT / 'supplement_plan.json').write_text(json.dumps({
        'plan': 'occlusion_blind_v2_supplement',
        'rule': '每段 20 帧，5%–95% 等间隔；时刻在看模型前固定',
        'why': '第一批 6 段中仅 1 段为板面素材，其余为炒锅/采访/流水线',
        'model_inference_performed': False,
        'sources': [{'bvid': b, 'title': t} for b, t in SOURCES],
        'frames': records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 12)
    per_sheet = 20
    for index, start in enumerate(range(0, len(records), per_sheet)):
        canvas = Image.new('RGB', (1250, 560), 'white')
        for slot, rec in enumerate(records[start:start + per_sheet]):
            image = Image.open(rec['file']).convert('RGB')
            image.thumbnail((245, 130))
            canvas.paste(image, ((slot % 5) * 250, (slot // 5) * 140))
            ImageDraw.Draw(canvas).text(((slot % 5) * 250 + 2, (slot // 5) * 140 + 128),
                                        '%s %.0fs' % (rec['source_bvid'][-6:], rec['seconds']),
                                        fill='black', font=font)
        canvas.save(OUT / ('sheet_%d.jpg' % (7 + index)), quality=88)
    print('total', len(records), 'sheets 7-', 7 + (len(records) - 1) // per_sheet)


if __name__ == '__main__':
    main()
