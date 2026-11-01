"""Resume the interrupted 29th corn training from its last checkpoint, then run the fixed evaluations.

Why: food_fullmask_corn_multiscale45_v1 was killed at epoch 8/45 when the previous session ended
(no python process, results.csv has 7 rows, status.json still says training). Training args are
restored by ultralytics from the run's own args.yaml, so resume continues to epoch 45.

After training, the same post-training artefacts the original script produces are generated
(presence patches, old/aligned val, corn validation groups), then status.json is marked completed
so finalize_corn_fullmask_training.py can run its report/kitchen/fit steps.
"""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ['YOLO_OFFLINE'] = 'True'
os.environ['YOLO_AUTOINSTALL'] = 'False'

from ultralytics import YOLO  # noqa: E402

B = Path(__file__).resolve().parent
E = B / 'food_fullmask_corn_multiscale45_v1'


def save(name, data):
    (E / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    last = E / 'training' / 'weights' / 'last.pt'
    if not last.exists():
        raise RuntimeError(f'missing checkpoint: {last}')
    status = json.loads((E / 'status.json').read_text(encoding='utf-8'))
    print('start status', status, flush=True)
    save('recovery.json', {
        'resumed_from': str(last), 'checkpoint_sha256': sha(last),
        'resumed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'reason': 'training process was killed with the previous session; resumed from last.pt',
        'note': 'optimizer/epoch restored by ultralytics resume; the custom size-count callback is not '
                're-attached on resume, so observed_train_sizes.json is not extended',
    })
    save('status.json', dict(stage='training_resumed', adopted=False))

    model = YOLO(str(last))
    model.train(resume=True)

    model = YOLO(str(E / 'training' / 'weights' / 'last.pt'))
    assert model.names == {0: 'meat', 1: 'vegetable'}, model.names
    print('training finished, running fixed evaluations', flush=True)
    save('status.json', dict(stage='evaluating', adopted=False))

    rows = json.loads((B / 'food_presence_patches_v3' / 'reviewed_manifest.json').read_text(encoding='utf-8'))['records']
    out = []
    for row in rows:
        prediction = model.predict(row['crop'], conf=.25, imgsz=640, rect=False, device=0, verbose=False)[0]
        out.append(dict(id=row['id'], reference=row['label'], source=row['source_group'],
                        predictions=[dict(category=model.names[int(q.cls[0])], confidence=float(q.conf[0]),
                                          box=q.xyxy[0].tolist()) for q in prediction.boxes]))
    save('after.json', out)

    yaml = E / 'data.yaml'
    for name, data in [('old_after', yaml), ('aligned_after', B / 'food_aligned_replay_v1' / 'aligned.yaml')]:
        metrics = model.val(data=str(data), imgsz=640, device=0, batch=8, workers=0, plots=False,
                            project=str(E), name=name, verbose=False)
        save(name + '.json', dict(map50=float(metrics.box.map50), map50_95=float(metrics.box.map)))
        print(name, 'done', flush=True)

    for group in ['old64', 'fishfruit30']:
        metrics = model.val(data=str(B / 'food_corn_validation_v1' / f'{group}.yaml'), imgsz=640, rect=False,
                            device=0, batch=8, workers=0, plots=False, project=str(E),
                            name='corn_' + group, verbose=False)
        save('corn_' + group + '.json', dict(map50=float(metrics.box.map50), map50_95=float(metrics.box.map)))
        print(group, 'done', flush=True)

    save('status.json', dict(stage='completed', adopted=False, default_unchanged=True,
                             resumed=True))
    print('COMPLETED', flush=True)


if __name__ == '__main__':
    main()
