"""板面标注队列：列出所有"人工标为有板"的帧，标出是否已有人工框，并生成编号对照表。

用途：板面标注是当前唯一的规模化解法，但需要人一张张画。这个脚本负责
  1) 把待标注帧列清楚（按来源分组、带编号）；
  2) 已标注的（存在 <图名>.board.json）自动跳过；
  3) 生成 contact sheet，编号与队列一一对应，画框时按编号对照即可。

用法：
  python board_annotation_queue.py [--out experiments/board_annotation_v1]
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
EXPERIMENTS = ['new_source_video_v1', 'new_source_video_v2', 'new_source_video_v3', 'new_source_video_v4']


def gather():
    items = []
    for name in EXPERIMENTS:
        directory = BASE / 'experiments' / name
        labels_file = directory / 'labels_blind.json'
        if not labels_file.exists():
            continue
        labels = json.loads(labels_file.read_text(encoding='utf-8'))['labels']
        for frame_name, entry in labels.items():
            if entry.get('board_present') != 'yes' or entry.get('usable') is False:
                continue
            frame_path = directory / 'frames' / frame_name
            if not frame_path.exists():
                continue
            sidecar = frame_path.with_suffix('.board.json')
            items.append({'source': name, 'frame': frame_name, 'path': str(frame_path),
                          'has_human_box': sidecar.exists(), 'sidecar': str(sidecar) if sidecar.exists() else None,
                          'source_bvid': frame_name.split('_')[0]})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='experiments/board_annotation_v1')
    args = parser.parse_args()
    out = BASE / args.out
    out.mkdir(parents=True, exist_ok=True)

    items = gather()
    pending = [item for item in items if not item['has_human_box']]
    payload = {
        'total_board_frames': len(items),
        'with_human_box': len(items) - len(pending),
        'pending': len(pending),
        'note': 'human boxes are stored next to the frame as <name>.board.json via set_board_box.py',
        'items': items,
    }
    (out / 'queue.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    font = ImageFont.truetype(str(FONT), 14)
    per_sheet = 12
    for start in range(0, len(items), per_sheet):
        canvas = Image.new('RGB', (1200, 540), 'white')
        for slot, item in enumerate(items[start:start + per_sheet]):
            image = Image.open(item['path']).convert('RGB')
            image.thumbnail((300, 300))
            canvas.paste(image, ((slot % 4) * 300, (slot // 4) * 180))
            index = start + slot
            mark = 'OK' if item['has_human_box'] else 'TODO'
            ImageDraw.Draw(canvas).text(((slot % 4) * 300 + 4, (slot // 4) * 180 + 4),
                                        f"#{index} {mark} {item['frame']}", fill='yellow', font=font)
        canvas.save(out / f'queue_sheet_{start // per_sheet + 1}.jpg')

    by_source = {}
    for item in items:
        by_source.setdefault(item['source'], {'total': 0, 'pending': 0})
        by_source[item['source']]['total'] += 1
        if not item['has_human_box']:
            by_source[item['source']]['pending'] += 1
    print(json.dumps({'total_board_frames': len(items), 'pending': len(pending), 'by_source': by_source},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
