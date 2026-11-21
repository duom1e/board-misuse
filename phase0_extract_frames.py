"""阶段 0-c：为遮挡盲测集 v2 抽帧（时刻先固定，标签后标）。

每段视频按 5%–95% 等间隔取 20 帧；时刻在跑任何模型之前写进 manifest。
输出：experiments/occlusion_phase0/frames/*.jpg + sampling_plan.json + 联系表
"""
import cv2
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
INBOX = BASE.parent / '待导入视频'
OUT = BASE / 'experiments' / 'occlusion_phase0'
SOURCES = [
    ('BV1684y1r7HX', '沉浸式切菜（15:19）'),
    ('BV19b411v7Wz', '酒店后厨探秘（3:58）'),
    ('BV1krWG6KEUV', '后厨学徒工作详解（5:27）'),
    ('BV1ce4y1o74e', '中央厨房净菜流水线（2:45）'),
    ('BV1Gz4y1y7UH', '大型酒店后厨（5:29）'),
    ('BV1Vc411b7ov', '米其林后厨管理（15:56）'),
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

    (OUT / 'sampling_plan.json').write_text(json.dumps({
        'plan': 'occlusion_blind_v2',
        'stage': 'frames_frozen_labels_pending',
        'rule': '每段 20 帧，5%–95% 等间隔；时刻在看模型前固定',
        'labels_required': ['board_present(no/yes/unclear)', 'occlusion_level(none/mild/severe)',
                            'visible_box', 'full_box(inferred)'],
        'model_inference_performed': False,
        'sources': [{'bvid': b, 'title': t} for b, t in SOURCES],
        'frames': records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 12)
    per_sheet = 20
    for start in range(0, len(records), per_sheet):
        canvas = Image.new('RGB', (1250, 560), 'white')
        for slot, rec in enumerate(records[start:start + per_sheet]):
            image = Image.open(rec['file']).convert('RGB')
            image.thumbnail((245, 130))
            canvas.paste(image, ((slot % 5) * 250, (slot // 5) * 140))
            ImageDraw.Draw(canvas).text(((slot % 5) * 250 + 2, (slot // 5) * 140 + 128),
                                        '%s %.0fs' % (rec['source_bvid'][-6:], rec['seconds']),
                                        fill='black', font=font)
        canvas.save(OUT / ('sheet_%d.jpg' % (start // per_sheet + 1)), quality=88)
    print('total frames', len(records), 'sheets', (len(records) + per_sheet - 1) // per_sheet)


if __name__ == '__main__':
    main()
