"""数据集管线冒烟训练：用 13 张人工框图跑一次极小训练，只验证"格式与流程能走通"。

这不是模型实验：样本 13 张（train 10 / val 3），训练出来的东西没有意义。
目的是尽早发现标注格式、路径、类别配置等工程问题。
"""
import json
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA = BASE / 'datasets' / 'board_seg_v2' / 'data.yaml'
OUT = BASE / 'experiments' / 'board_seg_smoke_train_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'purpose.json').write_text(json.dumps({
        'what': 'datasets/board_seg_v2 的格式与训练流程冒烟',
        'records': 13, 'train': 10, 'val': 3,
        'not_a_model': True,
        'note': '样本量远不足以训练可用检测器；只看 loss 是否下降、验证能否跑完、导出的权重能否加载',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    model = YOLO(str(BASE / 'yolo11n.pt'))
    model.train(data=str(DATA), epochs=20, imgsz=640, batch=4, device=0, workers=0,
                project=str(OUT), name='train', exist_ok=True, plots=False, val=True,
                seed=42, deterministic=True, amp=False, cache=False)
    metrics = YOLO(str(OUT / 'train' / 'weights' / 'best.pt')).val(
        data=str(DATA), imgsz=640, batch=4, device=0, workers=0, plots=False, verbose=False)
    summary = {
        'smoke_only': True,
        'epochs': 20,
        'val_map50': float(metrics.box.map50),
        'val_map50_95': float(metrics.box.map),
        'note': '13 张图上的数值无语义，只证明管线能跑通',
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
