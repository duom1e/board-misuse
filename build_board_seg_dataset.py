"""把人工板框汇总成"板面检测/分割"训练集（bbox 版），按来源视频分组切分防泄漏。

数据来源：各 <图名>.board.json（set_board_box.py 生成，归一化 0-1000）。
输出：datasets/board_seg_v1/{images,labels}/{train,val} + data.yaml + manifest.json

注意：这是**标注流水线的雏形**，样本量取决于人工已画多少框；不够就不要训练。
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'datasets' / 'board_seg_v1'
EXPERIMENTS = ['new_source_video_v1', 'new_source_video_v2', 'new_source_video_v3', 'new_source_video_v4']
VAL_GROUPS = {'BV1xW411f7zP'}  # 按来源视频整组留出，避免同源泄漏


def collect():
    records = []
    for name in EXPERIMENTS:
        directory = BASE / 'experiments' / name / 'frames'
        if not directory.exists():
            continue
        for sidecar in sorted(directory.glob('*.board.json')):
            payload = json.loads(sidecar.read_text(encoding='utf-8'))
            frame = Path(payload['image'])
            if not frame.exists():
                continue
            box = payload['board']
            sha = hashlib.sha256(frame.read_bytes()).hexdigest()
            records.append({'frame': frame, 'box': box, 'group': frame.name.split('_')[0],
                            'sha256': sha, 'experiment': name})
    # 同一 SHA 只保留一次
    seen = {}
    for record in records:
        seen.setdefault(record['sha256'], record)
    return list(seen.values())


def main():
    records = collect()
    if not records:
        print('还没有任何人工板框（*.board.json）。先用 set_board_box.py 标几张。')
        return 1
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ('train', 'val'):
        (OUT / 'images' / split).mkdir(parents=True)
        (OUT / 'labels' / split).mkdir(parents=True)

    manifest = []
    for record in records:
        split = 'val' if record['group'] in VAL_GROUPS else 'train'
        target = OUT / 'images' / split / record['frame'].name
        try:
            os.link(record['frame'], target)
        except OSError:
            shutil.copyfile(record['frame'], target)
        box = record['box']
        cx = (box[0] + box[2]) / 2 / 1000
        cy = (box[1] + box[3]) / 2 / 1000
        bw = (box[2] - box[0]) / 1000
        bh = (box[3] - box[1]) / 1000
        (OUT / 'labels' / split / (record['frame'].stem + '.txt')).write_text(
            f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n', encoding='utf-8')
        manifest.append({'frame': record['frame'].name, 'split': split, 'group': record['group'],
                         'box_normalized': box, 'image_sha256': record['sha256'],
                         'experiment': record['experiment']})

    (OUT / 'data.yaml').write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: board\n", encoding='utf-8')
    (OUT / 'manifest.json').write_text(json.dumps({
        'source': 'human boxes from set_board_box.py', 'format': 'yolo bbox, normalized',
        'group_split': {'val_groups': sorted(VAL_GROUPS), 'rule': 'split by source video, not by frame'},
        'records': manifest,
        'caveat': 'sample size is whatever has been annotated; do not train until val has >=3 frames from a held-out source',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    train_count = sum(1 for m in manifest if m['split'] == 'train')
    val_count = len(manifest) - train_count
    print(json.dumps({'records': len(manifest), 'train': train_count, 'val': val_count,
                      'groups': sorted({m['group'] for m in manifest})}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
