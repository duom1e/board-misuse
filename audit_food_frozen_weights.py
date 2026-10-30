"""Verify real parameter freezing; compare only parameters, not BN buffers."""
import json,torch
from pathlib import Path
from ultralytics import YOLO
B=Path(__file__).resolve().parent;E=B/'food_frozen_backbone_v1'
def main():
 before=dict(YOLO(str(B/'training_runs/foodseg_2cls/weights/best.pt')).model.named_parameters());after=dict(YOLO(str(E/'training/weights/last.pt')).model.named_parameters());assert before.keys()==after.keys()
 frozen=[];changed=[]
 for name,v in before.items():
  index=int(name.split('.')[1]);equal=torch.equal(v.detach().cpu(),after[name].detach().cpu())
  if index<10:
   assert equal,name;frozen.append(name)
  elif not equal:changed.append(name)
 assert frozen and changed
 (E/'parameter_audit.json').write_text(json.dumps(dict(unchanged_first10_parameter_tensors=len(frozen),changed_later_parameter_tensors=len(changed),scope='Parameter comparison after checkpoint load; BN running statistics are buffers, excluded. This does not claim all intermediate features remain identical.'),indent=2),encoding='utf-8');print(len(frozen),'frozen parameter tensors unchanged;',len(changed),'later tensors updated')
if __name__=='__main__':main()
