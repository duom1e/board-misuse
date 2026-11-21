"""错误继承（false persistence）对照：三种时序策略在"板离开画面"窗口上的表现。

窗口与 GT（人工确认切换时刻，见 window_*.jpg）：
  · BV1wS4y1T78o 74.0–86.5s：76.0s 前为"板在"，76.5s 起为"板不在"（装盆）
  · BV1dh41187N8 424.0–443.5s：427.5s 前为"板在"，之后转手持特写（板基本出画）

三种策略：
  A 单帧：score ≥ τ 即报存在
  B 5 帧保持（项目现有规则）：当前帧有框，且最近 5 帧每帧都有 IoU≥0.5 的框
  C 朴素继承（反面教材）：当前帧无框时，沿用最近一次框最多 K 帧

指标：板在时的召回、板不在时的**错误继承率**、以及切换后需要多少帧才"闭嘴"。
"""
import json
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'occlusion_phase0'
WEIGHTS = BASE / 'training_runs' / 'board_neg15_v1' / 'weights' / 'last.pt'
WINDOWS = [
    {'bvid': 'BV1wS4y1T78o', 'present_until': 76.0, 'note': '装盆开始'},
    {'bvid': 'BV1dh41187N8', 'present_until': 427.5, 'note': '转手持特写'},
]
CONF, IOU, K = 0.25, 0.5, 30


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main():
    model = YOLO(str(WEIGHTS))
    report = []
    for window in WINDOWS:
        frames = sorted((OUT / 'frames').glob('w_%s_*.jpg' % window['bvid']))
        detections = []
        for path in frames:
            seconds = float(path.stem.split('_')[-1].rstrip('s'))
            boxes = model.predict(str(path), imgsz=640, rect=True, conf=.05, device=0, verbose=False)[0].boxes
            pairs = [(b, float(s)) for b, s in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist())]
            pairs = [(b, s) for b, s in pairs if s >= CONF]
            detections.append({'file': path.name, 'seconds': seconds,
                               'box': pairs[0][0] if pairs else None,
                               'score': pairs[0][1] if pairs else 0.0})
        gt_present = [d['seconds'] <= window['present_until'] for d in detections]

        # A 单帧
        policy_a = [d['box'] is not None for d in detections]

        # B 5 帧保持
        policy_b = []
        for index, det in enumerate(detections):
            keep = False
            if det['box'] is not None and index >= 4:
                keep = all(detections[j]['box'] is not None
                           and iou(det['box'], detections[j]['box']) >= IOU
                           for j in range(index - 4, index))
            policy_b.append(keep)

        # C 朴素继承
        policy_c = []
        last_seen = None
        for index, det in enumerate(detections):
            if det['box'] is not None:
                last_seen = index
                policy_c.append(True)
            else:
                policy_c.append(last_seen is not None and index - last_seen <= K)

        # 切换后多久才"闭嘴"：首次出现连续 3 帧判否的帧号；若从未闭嘴则为 None
        def clear_info(policy):
            after = [(index, p) for index, (p, keep) in enumerate(zip(policy, gt_present)) if not keep]
            if not after:
                return {'frames_flagged': 0, 'first_quiet_seconds_after_cut': None}
            flagged = sum(1 for _, p in after if p)
            quiet_start = None
            for position in range(len(after) - 2):
                if not after[position][1] and not after[position + 1][1] and not after[position + 2][1]:
                    quiet_start = after[position][0]
                    break
            cut_index = after[0][0]
            return {'frames_flagged': flagged,
                    'first_quiet_seconds_after_cut': (None if quiet_start is None
                                                      else round((quiet_start - cut_index) * 0.5, 1))}

        def stats(policy):
            present_hits = sum(1 for p, keep in zip(policy, gt_present) if keep and p)
            present_n = sum(gt_present)
            absent_flags = sum(1 for p, keep in zip(policy, gt_present) if (not keep) and p)
            absent_n = len(gt_present) - present_n
            info = clear_info(policy)
            return {'present_recall': '%d/%d' % (present_hits, present_n),
                    'false_persistence': '%d/%d' % (absent_flags, absent_n),
                    'frames_flagged_after_cut': info['frames_flagged'],
                    'first_quiet_seconds_after_cut': info['first_quiet_seconds_after_cut']}

        report.append({'bvid': window['bvid'], 'frames': len(detections),
                       'present_until': window['present_until'],
                       'A_single_frame': stats(policy_a),
                       'B_five_frame': stats(policy_b),
                       'C_naive_inherit': stats(policy_c),
                       'per_frame': [{'seconds': d['seconds'], 'gt': g, 'A': a, 'B': b, 'C': c,
                                      'score': round(d['score'], 3)}
                                     for d, g, a, b, c in zip(detections, gt_present, policy_a, policy_b, policy_c)]})

    (OUT / 'persistence_eval.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in report:
        print(row['bvid'], 'frames', row['frames'])
        for name in ('A_single_frame', 'B_five_frame', 'C_naive_inherit'):
            print('   ', name, row[name])


if __name__ == '__main__':
    main()
