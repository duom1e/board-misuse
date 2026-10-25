"""Manual board annotations: explicit review, no automatic negative labels."""
import hashlib
import json
import uuid
from pathlib import Path
from PIL import Image, ImageOps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_image(source, root):
    source = Path(source).resolve()
    folder = Path(root) / ('case_' + uuid.uuid4().hex)
    folder.mkdir(parents=True, exist_ok=False)
    with Image.open(source) as im:
        normalized = ImageOps.exif_transpose(im).convert('RGB')
        normalized.save(folder / 'image.png')
    metadata = {'original_path': str(source), 'original_sha256': sha(source),
                'normalized_sha256': sha(folder / 'image.png'),
                'normalization': 'EXIF transpose; RGB; lossless PNG',
                'size': list(normalized.size)}
    (folder / 'source.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return folder, normalized, metadata


def normalized_box(start, end, width, height):
    x1, x2 = sorted((max(0, min(width, start[0])), max(0, min(width, end[0]))))
    y1, y2 = sorted((max(0, min(height, start[1])), max(0, min(height, end[1]))))
    if x2-x1 < 3 or y2-y1 < 3:
        raise ValueError('框太小，请重新拖动。')
    return [x1/width, y1/height, x2/width, y2/height]


def export_annotation(image_path, metadata, boxes, group, decision, root):
    if not group.strip():
        raise ValueError('请填写来源组；同一视频或连拍使用相同的组名。')
    if decision not in ('all_boards_boxed', 'confirmed_no_board'):
        raise ValueError('请明确选择整图复核结论。')
    if decision == 'all_boards_boxed' and not boxes:
        raise ValueError('请框出至少一个案板。')
    if decision == 'confirmed_no_board' and boxes:
        raise ValueError('当前有案板框，不能导出无案板负样本。')
    for box in boxes:
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError('案板框坐标无效。')
    root = Path(root)
    # A separate transaction folder keeps partial writes out of any future dataset scan.
    folder = root / ('manual_' + uuid.uuid4().hex)
    folder.mkdir(parents=True, exist_ok=False)
    with Image.open(image_path) as im:
        im.save(folder / 'image.png')
    labels = []
    for x1, y1, x2, y2 in boxes:
        labels.append(f'0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}')
    (folder / 'image.txt').write_text('\n'.join(labels) + ('\n' if labels else ''), encoding='utf-8')
    record = {**metadata, 'source_group': group.strip(), 'review_decision': decision,
              'boxes_normalized': boxes, 'classes': ['cutting_board'],
              'reviewer': 'user_manual_GUI', 'data_status': 'unassigned_development',
              'note': 'Manual board boxes only; no food or violation truth. Freshness and scene independence not verified.',
              'exported_image_sha256': sha(folder / 'image.png')}
    (folder / 'metadata.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    (folder / 'COMPLETE').write_text('Ready for later grouped dataset review; not automatically added to training.\n', encoding='utf-8')
    return folder
