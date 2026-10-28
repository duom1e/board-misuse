"""One command: real video -> sampled VLM observations -> offline evidence page."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,required=True);p.add_argument('--interval',type=float,default=10);p.add_argument('--output',type=Path,required=True);p.add_argument('--demo-allowed',choices=['vegetable','meat','both'],help='Explicit hypothetical policy, never verified site rules');a=p.parse_args()
 if not a.video.is_file():p.error('Video missing')
 if a.output.exists():p.error('Output must be new; inspect previous status before retrying')
 # Video reader CLI validates exact frame schedule before model loading.
 if a.start<0 or a.end<a.start or a.interval<=0:p.error('Invalid time range/interval')
 a.output.mkdir(parents=True)
 def save(stage,**extra):(a.output/'pipeline_status.json').write_text(json.dumps(dict(stage=stage,**extra),ensure_ascii=False,indent=2),encoding='utf-8')
 plan=dict(video=str(a.video.resolve()),video_sha256=hashlib.sha256(a.video.read_bytes()).hexdigest(),start=a.start,end=a.end,interval=a.interval,policy='unconfigured' if not a.demo_allowed else 'explicit_demo_only',demo_allowed=a.demo_allowed,training_performed=False)
 (a.output/'pipeline_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
 commands=[('recognition',[sys.executable,str(B/'review_board_video.py'),str(a.video),'--start',str(a.start),'--end',str(a.end),'--interval',str(a.interval),'--food-categories','--food-context','--output-dir',str(a.output/'recognition')]),('evidence',[sys.executable,str(B/'build_misuse_demo.py'),str(a.output/'recognition/results.json'),'--allowed',a.demo_allowed or 'unconfigured','--output',str(a.output/'evidence')])]
 for stage,cmd in commands:
  save(stage)
  with (a.output/f'{stage}.log').open('w',encoding='utf-8') as log:
   result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:save('failed',failed_stage=stage,exit_code=result.returncode);raise SystemExit(result.returncode)
 save('video_binding')
 with (a.output/'video_binding.log').open('w',encoding='utf-8') as log:
  result=subprocess.run([sys.executable,str(B/'add_video_timeline.py'),str(a.output)],stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:save('failed',failed_stage='video_binding',exit_code=result.returncode);raise SystemExit(result.returncode)
 save('completed',page=str((a.output/'evidence/index.html').resolve()),training_performed=False)
 print(a.output/'evidence/index.html')
if __name__=='__main__':main()
