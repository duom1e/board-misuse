import json,hashlib,subprocess,sys
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'food_batch36_v2'
def main():
 rows=json.loads((E/'results.json').read_text(encoding='utf-8'))['rows'];assert len(rows)==36;groups={};crops=[];issues=[]
 for i,r in enumerate(rows):
  group=groups.setdefault(r['source'],dict(samples=0,valid=0,board_frames=0,food_candidates=0,categories=Counter(),locations=Counter()));group['samples']+=1;group['valid']+=r['validation']['valid'];group['board_frames']+=bool(r['validation']['normalized_boxes'] and r['validation']['normalized_boxes']['boards']);group['food_candidates']+=len(r['regions'])
  if not r['validation']['valid']:issues.append(dict(index=i,source=r['source'],seconds=r['frame_seconds'],reason='invalid_response',errors=r['validation']['errors']))
  for j,q in enumerate(r['regions']):
   group['locations'][q['location']]+=1
   if q.get('category_inference_performed'):group['categories'][q['food_category']]+=1
   if 'category_crop_xyxy' in q:
    path=E/f'{i:02}_{j:02}_crop.jpg';assert path.is_file();crops.append(dict(index=i,region=j,source=r['source'],seconds=r['frame_seconds'],category=q['food_category'],crop=str(path)))
   if q['location']=='uncertain' or q.get('category_inference_performed') and q['food_category']=='unknown':issues.append(dict(index=i,region=j,source=r['source'],seconds=r['frame_seconds'],reason=q['location'] if q['location']=='uncertain' else 'category_abstention',box=q['box']))
 for start in range(0,36,12):
  sheet=Image.new('RGB',(1280,1000));d=ImageDraw.Draw(sheet)
  for j,r in enumerate(rows[start:start+12]):
   im=Image.open(E/f'{start+j:02}_review.jpg');im.thumbnail((320,260));x=j%4*320;y=j//4*330;sheet.paste(im,(x,y));d.text((x,y+265),f"{start+j} {r['source']}\n{r['frame_seconds']:.2f}s valid={r['validation']['valid']}",fill='white')
  sheet.save(E/f'scene_page_{start//12}.jpg')
 for start in range(0,len(crops),12):
  sheet=Image.new('RGB',(1280,1000));d=ImageDraw.Draw(sheet)
  for j,r in enumerate(crops[start:start+12]):
   im=Image.open(r['crop']);im.thumbnail((310,260));x=j%4*320;y=j//4*330;sheet.paste(im,(x,y));d.text((x,y+265),f"{r['index']}:{r['region']} {r['category']}\n{r['seconds']:.2f}s",fill='white')
  sheet.save(E/f'crop_page_{start//12}.jpg')
 audit=dict(groups=groups,crops=crops,issues=issues,samples=36,training_performed=False,independent_accuracy=False,source_results_sha256=hashlib.sha256((E/'results.json').read_bytes()).hexdigest());(E/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(groups=groups,crops=len(crops),issues=len(issues)),ensure_ascii=False))
 subprocess.run([sys.executable,str(B/'build_misuse_demo.py'),str(E/'results.json'),'--output',str(E/'evidence')],check=True)
if __name__=='__main__':main()
