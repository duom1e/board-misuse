"""Sample video food proposals with independent category opinions; no board policy."""
import argparse,json,hashlib,os,html
from pathlib import Path
os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('video',type=Path);p.add_argument('--weights',type=Path,required=True);p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,required=True);p.add_argument('--interval',type=float,default=10);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if not a.video.is_file() or not a.weights.is_file():p.error('Video and local weights must exist')
 if a.output.exists():p.error('Output must be new')
 import cv2
 from PIL import Image,ImageDraw
 from video_sample_plan import plan_frames
 cap=cv2.VideoCapture(str(a.video))
 try:samples=plan_frames(a.start,a.end,a.interval,cap.get(5),int(cap.get(7)))
 except ValueError as e:cap.release();p.error(str(e))
 if len(samples)>20:cap.release();p.error('At most20 samples for this diagnostic entrypoint')
 a.output.mkdir(parents=True)
 def save(n,x):(a.output/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
 save('plan.json',dict(video=str(a.video.resolve()),video_sha256=sha(a.video),weights=str(a.weights.resolve()),weights_sha256=sha(a.weights),samples=samples,conf=.25,imgsz=640,rect=False,scope='Food candidates only; no board detection, location/contact inference or compliance. Same frozen VLM opinions, original labels preserved.'))
 save('status.json',dict(stage='loading'))
 from ultralytics import YOLO
 from region_vlm_runtime import LocalRegionModel
 from food_category_review import FOOD_CATEGORY_PROMPT,review_board_foods
 detector=YOLO(str(a.weights));model=LocalRegionModel();rows=[];cards=[]
 for i,s in enumerate(samples):
  row=dict(**s,status='needs_review',compliance_determined=False)
  try:
   cap.set(1,s['frame_index']);ok,f=cap.read()
   if not ok or int(cap.get(1))-1!=s['frame_index']:raise ValueError('Exact frame decode failed')
   imagepath=a.output/f'{i:03d}.jpg';cv2.imwrite(str(imagepath),f);im=Image.open(imagepath).convert('RGB');row.update(image_sha256=sha(imagepath),predictions=[])
   pred=detector.predict(str(imagepath),imgsz=640,rect=False,conf=.25,device='cpu',verbose=False)[0];overlay_image=im.copy();draw=ImageDraw.Draw(overlay_image)
   for j,q in enumerate(pred.boxes):
    box=q.xyxy[0].tolist();original=detector.names[int(q.cls[0])]
    review=review_board_foods(im,[dict(box=box,location='board_candidate')],lambda crop:model.predict(crop,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12))[0];review.pop('location')
    row['predictions'].append(dict(box=box,original_category=original,detector_confidence=float(q.conf[0]),review=review,original_category_preserved=True,board_relation_evaluated=False))
    draw.rectangle(box,outline='orange',width=3);draw.text((box[0],box[1]),str(j+1),fill='orange')
   overlay=f'{i:03d}_review.jpg';overlay_image.save(a.output/overlay)
   labels=''.join(f'<li>候选{j+1}：检测 {html.escape(q["original_category"])}；复核 {html.escape(q["review"]["food_category"])}</li>' for j,q in enumerate(row['predictions'])) or '<li>没有食材候选，不代表没有食材或合规。</li>'
   cards.append(f'<article><h2>{s["frame_seconds"]:.3f}秒</h2><img src="{overlay}"><ul>{labels}</ul></article>')
  except Exception as exc:row['error']=repr(exc);cards.append(f'<article><h2>采样{i+1}失败</h2><p>证据不足，详见JSON。</p></article>')
  rows.append(row);save('partial_results.json',dict(rows=rows));print(i+1,len(samples),len(row.get('predictions',[])),flush=True)
 cap.release();save('results.json',dict(rows=rows,training_performed=False,default_changed=False,compliance_determined=False))
 (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>检测候选与类别复核</title><style>body{font:16px sans-serif;background:#edf2f5;margin:24px}article{background:white;padding:20px;margin:20px auto;max-width:1000px}img{max-width:100%}</style><h1>视频食材候选与类别复核</h1><p>橙框仅为食材候选，未检测案板用途或接触。两种类别保留；一致也可能出错，unknown仅为弃答。固定采样，未跟踪，不能将多个采样当多个违规事件。</p>'+''.join(cards),encoding='utf-8')
 save('status.json',dict(stage='completed',samples=len(rows),failed_samples=sum('error' in r for r in rows)))
if __name__=='__main__':main()
