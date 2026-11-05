"""把人工板框汇总成"案板物体检测"训练集（bbox 版），按来源视频分组切分防泄漏。

数据来源：各 <图名>.board.json（set_board_box.py 生成，归一化 0-1000）。
输出：--output-dir 指定的新目录；不覆盖已有目录，不生成分割掩膜。

注意：这是**标注流水线的雏形**，样本量取决于人工已画多少框；不够就不要训练。
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from review_combined import load_human_annotation

BASE = Path(__file__).resolve().parent
OUT = BASE / 'datasets' / 'board_seg_v1'
EXPERIMENTS = ['new_source_video_v1', 'new_source_video_v2', 'new_source_video_v3', 'new_source_video_v4', 'new_source_video_v5', 'new_source_video_v8']
VAL_GROUPS = {'BV1pu411U7Zh'}  # 按来源视频整组留出，避免同源泄漏


def collect():
    records = []
    for name in EXPERIMENTS:
        directory = BASE / 'experiments' / name / 'frames'
        if not directory.exists():
            continue
        for sidecar in sorted(directory.glob('*.board.json')):
            payload = json.loads(sidecar.read_text(encoding='utf-8'))
            frame = Path(payload['image'])
            if not frame.is_file():
                raise ValueError(f'Missing image: {frame}')
            if not payload.get('image_sha256'):
                raise ValueError(f'Training annotation needs image_sha256: {sidecar}')
            box = load_human_annotation(sidecar, frame)
            sha = hashlib.sha256(frame.read_bytes()).hexdigest()
            records.append({'frame': frame, 'box': box, 'group': frame.name.split('_')[0],
                            'sha256': sha, 'experiment': name})
    # 同一 SHA 只保留一次
    seen = {}
    for record in records:
        previous = seen.get(record['sha256'])
        if previous and (previous['box'] != record['box'] or previous['group'] != record['group']):
            raise ValueError('Conflicting annotation or source group for duplicate image')
        seen.setdefault(record['sha256'], record)
    return list(seen.values())


def main():
    global OUT
    parser = argparse.ArgumentParser(description='Build a new immutable bbox dataset version')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if OUT.exists():
        parser.error('Output already exists; choose a new version directory')
    records = collect()
    names = [r['frame'].stem for r in records]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate image stems would overwrite labels')
    if not records:
        print('还没有任何人工板框（*.board.json）。先用 set_board_box.py 标几张。')
        return 1
    OUT.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val'):
        (OUT / 'images' / split).mkdir(parents=True)
        (OUT / 'labels' / split).mkdir(parents=True)

    manifest = []
    for record in records:
        split = 'val' if record['group'] in VAL_GROUPS else 'train'
        target = OUT / 'images' / split / record['frame'].name
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
