from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments/roast_source_check_20261111'

def main():
    OUT.mkdir(exist_ok=False)
    cases = []
    for frame, top, bottom in [('00', 305, 640), ('04', 204, 602), ('11', 232, 616)]:
        path = BASE / f'roast_screen_v1/BV1aR4y1Z7FR_{frame}.jpg'
        cases.append({'image': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'box': [0, top, 480, bottom]})
    plan = {'source': 'BV1aR4y1Z7FR', 'cases': cases, 'annotation': 'Single reviewer approximate visible object extent; occluded and clipped borders. Same video, not independent samples.', 'training_allowed': False, 'confidence': 0.5, 'iou': 0.5}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    models = {
        'default': BASE / 'training_runs/board_v2_real/weights/best.pt',
        'real5': BASE / 'experiments/synthetic_board_pilot_v1/real_run/weights/last.pt',
        'roast5': BASE / 'experiments/roast_mid_board_v1/mixed_run/weights/last.pt',
    }
    summary = []
    for tag, weight in models.items():
        model = YOLO(str(weight))
        rows = []
        canvas = Image.new('RGB', (480 * 3, 680), 'white')
        for index, case in enumerate(cases):
            prediction = model.predict(case['image'], imgsz=960, rect=True, conf=0.01, device=0, verbose=False)[0]
            boxes = []
            gt = case['box']
            picture = Image.open(case['image']).convert('RGB')
            draw = ImageDraw.Draw(picture)
            draw.rectangle(gt, outline='lime', width=3)
            for box, confidence in zip(prediction.boxes.xyxy.cpu().tolist(), prediction.boxes.conf.cpu().tolist()):
                overlap = max(0, min(box[2], gt[2])-max(box[0], gt[0])) * max(0, min(box[3], gt[3])-max(box[1], gt[1]))
                union = (box[2]-box[0])*(box[3]-box[1]) + (gt[2]-gt[0])*(gt[3]-gt[1]) - overlap
                boxes.append({'box': box, 'confidence': confidence, 'iou': overlap / union})
                if confidence >= 0.5:
                    draw.rectangle(box, outline='red', width=3)
            canvas.paste(picture, (index*480, 30))
            rows.append({**case, 'predictions': boxes})
        for threshold in [0.01, 0.1, 0.25, 0.5]:
            hits = sum(any(b['confidence'] >= threshold and b['iou'] >= 0.5 for b in row['predictions']) for row in rows)
            summary.append({'model': tag, 'threshold': threshold, 'hits': hits, 'total': len(rows)})
        canvas.save(OUT / f'{tag}_review.jpg')
        (OUT / f'{tag}.json').write_text(json.dumps({'weight_sha256': hashlib.sha256(weight.read_bytes()).hexdigest(), 'rows': rows}, indent=2), encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))

if __name__ == '__main__':
    main()
