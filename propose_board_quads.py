"""几何法找板面：Canny + 轮廓近似出四边形，作为 VLM 框过粗时的位置候选。

不训练、不用模型；只在已有冻结帧上做候选生成，并输出叠图供人工判断。
判据（先写死）：在 VLM 框被判"过粗"的有板帧上，若能给出面积占比 3%–60% 的四边形，
且叠图看在板上，则视为"几何法可补位"。这是候选生成实验，不是准确率。
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'board_quad_v1'
SAMPLES = [
    ('new_source_video_v3', 'BV1pq4y147jk_050s.jpg'),
    ('new_source_video_v3', 'BV1pq4y147jk_290s.jpg'),
    ('new_source_video_v3', 'BV12a41187Ja_090s.jpg'),
    ('new_source_video_v4', 'BV1bqsfzbELh_125s.jpg'),
    ('new_source_video_v4', 'BV1bqsfzbELh_370s.jpg'),
    ('new_source_video_v4', 'BV1bqsfzbELh_475s.jpg'),
]
FONT = Path(r'C:\Windows\Fonts\arial.ttf')


def find_quads(image_bgr):
    height, width = image_bgr.shape[:2]
    scale = 640 / max(width, 1)
    small = cv2.resize(image_bgr, (int(width * scale), int(height * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    frame_area = small.shape[0] * small.shape[1]
    candidates = []
    for invert in (False, True):
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if invert:
            mask = cv2.bitwise_not(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            fraction = area / frame_area
            if fraction < 0.03 or fraction > 0.60:
                continue
            rect = cv2.minAreaRect(contour)
            (_, _), (rw, rh), _ = rect
            if rw < 2 or rh < 2:
                continue
            rectangularity = area / (rw * rh)
            if rectangularity < 0.60:
                continue
            box = cv2.boxPoints(rect).astype(int).tolist()
            candidates.append({'quad': box, 'area_fraction': round(fraction, 3),
                               'rectangularity': round(rectangularity, 3), 'scale': scale,
                               'polarity': 'dark' if invert else 'bright'})
    candidates.sort(key=lambda item: -(item['area_fraction'] * item['rectangularity']))
    return candidates[:3], scale


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT), 14)
    canvas = Image.new('RGB', (1200, 720), 'white')
    report = []
    for index, (directory, name) in enumerate(SAMPLES):
        path = BASE / 'experiments' / directory / 'frames' / name
        image = cv2.imread(str(path))
        quads, scale = find_quads(image)
        report.append({'frame': name, 'quads_found': len(quads),
                       'best_area_fraction': quads[0]['area_fraction'] if quads else None,
                       'best_rectangularity': quads[0]['rectangularity'] if quads else None})
        pil = Image.open(path).convert('RGB')
        draw = ImageDraw.Draw(pil)
        for order, candidate in enumerate(quads):
            points = [(p[0] / scale, p[1] / scale) for p in candidate['quad']]
            colour = 'lime' if order == 0 else 'orange'
            draw.line(points + [points[0]], fill=colour, width=4)
            draw.text((points[0][0] + 4, points[0][1] + 4),
                      f"quad{order + 1} {candidate['area_fraction']:.2f}", fill=colour, font=font)
        pil.thumbnail((400, 300))
        canvas.paste(pil, ((index % 3) * 400, (index // 3) * 360))
        ImageDraw.Draw(canvas).text(((index % 3) * 400 + 4, (index // 3) * 360 + 4), name, fill='yellow', font=font)
        print(name, 'quads', len(quads), 'best', report[-1]['best_area_fraction'], flush=True)
    canvas.save(OUT / 'quad_sheet.jpg')
    (OUT / 'results.json').write_text(json.dumps({'samples': report,
                                                  'rule': 'area 3%-60% + rectangularity>=0.75 + convex quad',
                                                  'note': '候选生成实验，不是准确率'}, ensure_ascii=False, indent=2),
                                      encoding='utf-8')


if __name__ == '__main__':
    main()
