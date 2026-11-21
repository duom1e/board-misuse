"""阶段 0-c3：第三批抽帧（2 段刀工素材），联系表编号 10–11。"""
import cv2
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
INBOX = BASE.parent / '待导入视频'
OUT = BASE / 'experiments' / 'occlusion_phase0'
SOURCES = [
    ('BV1ux411g7tW', '餐厅主厨展示基本刀工（8:13）'),
    ('BV1nm4y1Z7qs', '大厨切土豆丝（1:31）'),
]
PER_VIDEO = 20


def main():
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

    (OUT / 'supplement2_plan.json').write_text(json.dumps({
        'plan': 'occlusion_blind_v2_supplement2',
        'rule': '每段 20 帧，5%–95% 等间隔；时刻在看模型前固定',
        'sources': [{'bvid': b, 'title': t} for b, t in SOURCES],
        'model_inference_performed': False,
        'frames': records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 12)
    for index, start in enumerate(range(0, len(records), 20)):
        canvas = Image.new('RGB', (1250, 560), 'white')
        for slot, rec in enumerate(records[start:start + 20]):
            image = Image.open(rec['file']).convert('RGB')
            image.thumbnail((245, 130))
            canvas.paste(image, ((slot % 5) * 250, (slot // 5) * 140))
            ImageDraw.Draw(canvas).text(((slot % 5) * 250 + 2, (slot // 5) * 140 + 128),
                                        '%s %.0fs' % (rec['source_bvid'][-6:], rec['seconds']),
                                        fill='black', font=font)
        canvas.save(OUT / ('sheet_%d.jpg' % (10 + index)), quality=88)
    print('total', len(records))


if __name__ == '__main__':
    main()
