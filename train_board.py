"""Train on real, labeled YOLO detection data. No dataset is bundled."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to labeled dataset YAML')
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if not Path(args.data).is_file():
        parser.error('Dataset YAML does not exist')
    from ultralytics import YOLO
    YOLO(args.model).train(data=str(Path(args.data).resolve()), epochs=args.epochs,
                          batch=args.batch, imgsz=640, device=0, workers=0,
                          project=str(Path(__file__).parent / 'training_runs'))

if __name__ == '__main__':
    main()
