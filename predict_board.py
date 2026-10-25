"""Predict cutting boards in an image/video with the trained baseline."""
import argparse
import os
from pathlib import Path

BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work'/'yolo-config')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',help='Local image/video file or folder')
    parser.add_argument('--weights',default=str(BASE/'training_runs'/'board_baseline'/'weights'/'best.pt'))
    parser.add_argument('--conf',type=float,default=0.25)
    args=parser.parse_args()
    if not Path(args.source).exists():
        parser.error('Source file or directory does not exist')
    if not Path(args.weights).is_file():
        parser.error('Trained weights do not exist')
    from ultralytics import YOLO
    results=YOLO(args.weights).predict(source=args.source,device=0,imgsz=640,conf=args.conf,
                    save=True,stream=True,project=str(BASE/'predictions'),name='board')
    for _ in results:
        pass

if __name__=='__main__':
    main()
