"""Reuse locked kitchen references for the taxonomy-aligned candidate."""
import shutil
from pathlib import Path
import evaluate_kitchen_source_validation as evaluation
B=Path(__file__).resolve().parent
if __name__=='__main__':
 e=B/'meat_operation_30epochs_kitchen_v1';e.mkdir(exist_ok=False)
 shutil.copyfile(B/'kitchen_source_validation_v1/manifest.json',e/'manifest.json')
 evaluation.E=e
 evaluation.MODELS={'original':B/'training_runs/foodseg_2cls/weights/best.pt','aligned':B/'food_meat_operation_30epochs_v1/training/weights/last.pt'}
 evaluation.main()
