"""盲测 v1：两段从未参与任何实验的厨房视频。

流程（先写判据再看模型）：
  1) 固定抽样时刻（每段 10 帧，等间隔）；
  2) 导出帧并人工盲标"有没有砧板"（不看任何模型输出）；
  3) 用候选模型 board_neg15_v1 跑一次，conf .25 / .5 各算一次；
  4) 记录命中、误报、漏检、位置可用。

本脚本只负责抽帧与建表；标签由人工写入 labels_blind.json 后再跑评估。
"""
import cv2
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
INBOX = BASE.parent / '待导入视频'
OUT = BASE / 'experiments' / 'blind_test_v1'
SOURCES = {
    'BV1Mg411U7SV': {'title': '酒店后厨工作流程（2:35）', 'count': 10},
    'BV1Zq4y1j7mm': {'title': '真实餐厅厨房实拍（3:44）', 'count': 10},
}


def main():
    (OUT / 'frames').mkdir(parents=True, exist_ok=True)
    records = []
    for bvid, spec in SOURCES.items():
        video = next(INBOX.glob(f'*{bvid}*.mp4'))
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps
        step = duration / (spec['count'] + 1)
        for k in range(1, spec['count'] + 1):
            seconds = round(step * k, 2)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
            ok, frame = capture.read()
            if not ok:
                continue
            name = '%s_%05.1fs.jpg' % (bvid, seconds)
            path = OUT / 'frames' / name
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            records.append({'source_bvid': bvid, 'video_title': spec['title'],
                            'seconds': seconds, 'file': str(path),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        capture.release()
        print(bvid, 'duration', round(duration, 1), 'frames', len([r for r in records if r['source_bvid'] == bvid]))
    (OUT / 'frames_manifest.json').write_text(json.dumps({
        'plan': 'blind_test_v1',
        'stage': 'frames_frozen_labels_pending',
        'rule': '抽样时刻在看模型前固定；标签在跑模型前写入；位置可用按框面积 <0.6 且非整幅',
        'model_inference_performed': False,
        'frames': records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('total', len(records))


if __name__ == '__main__':
    main()
