"""人体区域上游屏蔽探针：用 COCO person 检测先框出人，再判断"误检是否会被挡掉"。

判据先写死（experiments/person_suppression_v1/decision_rule.json）：
  主指标 = 人体区域内被标记的误检中，落进 person 框的比例（≥0.75 视为有用）
  护栏   = 非人体区域的食物框被 person 框挡掉的比例（≤0.10）
  不训练、不改默认模型；只用已检查过的 4 张人体区域图。
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'person_suppression_v1'
HUMAN = BASE / 'human_region_regression_v1' / 'manifest.json'


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def main():
    global OUT

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None, help='食物模型权重；默认用注册表 foodseg')
    parser.add_argument('--tag', default='default', help='输出子目录标记')
    args = parser.parse_args()
    OUT = OUT / args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'decision_rule.json').write_text(json.dumps({
        'question': '在食物模型之前加一层 COCO person 检测（只屏蔽人物区域），能否挡掉人体误检而不伤食物召回？',
        'data': 'human_region_regression_v1 的 4 张已检查图（2 张人脸、2 张手部内部区域）',
        'frozen_before_run': True,
        'primary': '人体区域内被标记的 meat/vegetable 误检中，中心落入 person 框的比例（≥0.75 视为有用）',
        'guard': '非人体区域的食物框被 person 框挡掉的比例（≤0.10）',
        'not_doing': ['不训练', '不改默认模型', '不据此调阈值'],
        'limits': ['样本仅 4 张图', 'person 检测本身也有漏检', '手工标注的人体区域是目检，不是专家框'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    registry = json.loads((BASE / 'active_models.json').read_text(encoding='utf-8'))
    weights = args.weights or registry['foodseg']['weights']
    food = YOLO(weights)
    person = YOLO(str(BASE / 'yolo11n.pt'))

    manifest = json.loads(HUMAN.read_text(encoding='utf-8'))['records']
    rows = []
    flagged_total = flagged_suppressed = 0
    other_total = other_suppressed = 0
    for record in manifest:
        image = record['image']
        region = record['region']
        food_result = food.predict(source=image, conf=.25, imgsz=640, rect=False, device=0, verbose=False)[0]
        person_result = person.predict(source=image, conf=.25, imgsz=640, device=0, classes=[0], verbose=False)[0]
        person_boxes = person_result.boxes.xyxy.cpu().tolist() if person_result.boxes is not None and len(person_result.boxes) else []

        flagged, outside = [], []
        if food_result.boxes is not None and len(food_result.boxes):
            for box, conf, cls in zip(food_result.boxes.xyxy.cpu().tolist(),
                                      food_result.boxes.conf.cpu().tolist(),
                                      food_result.boxes.cls.cpu().tolist()):
                entry = {'box': [round(float(v), 1) for v in box], 'category': food.names[int(cls)],
                         'confidence': round(float(conf), 3),
                         'in_human_region': inside(center(box), region)}
                entry['covered_by_person'] = any(inside(center(box), p) for p in person_boxes)
                (flagged if entry['in_human_region'] else outside).append(entry)
        for entry in flagged:
            flagged_total += 1
            flagged_suppressed += entry['covered_by_person']
        for entry in outside:
            other_total += 1
            other_suppressed += entry['covered_by_person']
        rows.append({'index': record['index'], 'kind': record['kind'], 'image': image,
                     'person_boxes': [[round(float(v), 1) for v in b] for b in person_boxes],
                     'human_region_predictions': flagged, 'other_predictions': outside})

    summary = {
        'images': len(rows),
        'person_detected_images': sum(1 for r in rows if r['person_boxes']),
        'flagged_in_human_region': flagged_total,
        'flagged_would_be_suppressed': flagged_suppressed,
        'suppression_rate': round(flagged_suppressed / flagged_total, 3) if flagged_total else None,
        'other_food_boxes': other_total,
        'other_would_be_suppressed': other_suppressed,
        'collateral_rate': round(other_suppressed / other_total, 3) if other_total else None,
        'weights': weights,
        'tag': args.tag,
        'note': '探针，不是准确率；样本 4 张图',
    }
    (OUT / 'results.json').write_text(json.dumps({'summary': summary, 'rows': rows},
                                                 ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
