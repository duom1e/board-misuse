import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'meat_operation_replay_v1'
SPECS=[(1,[[105,330,305,400]]),(4,[[65,300,310,365],[90,345,330,395]]),(7,[[75,270,250,375]]),(9,[[105,330,310,385]])]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=False);s=json.loads((B/'raw_meat_domain_sources_v1/sources.json').read_text(encoding='utf-8'))[0];records=[]
 for i,boxes in SPECS:
  f=s['frames'][i];assert sha(f['image'])==f['sha256'];roi=[0,218,360,400];im=Image.open(f['image']).crop(roi);p=E/'images/train'/f'operation_{i}.jpg';p.parent.mkdir(parents=True,exist_ok=True);im.save(p,quality=95);local=[[a,b-218,c,d-218] for a,b,c,d in boxes];lp=E/'labels/train'/f'operation_{i}.txt';lp.parent.mkdir(parents=True,exist_ok=True);lp.write_text(''.join(f'0 {(a+c)/720:.8f} {(b+d)/364:.8f} {(c-a)/360:.8f} {(d-b)/182:.8f}\n' for a,b,c,d in local),encoding='utf-8')
  records.append(dict(image=str(p),label=str(lp),image_sha256=sha(p),label_sha256=sha(lp),source=s['id'],source_frame=f,roi=roi,boxes=local,review_status='pending_overlay',scope='Low-resolution video content ROI; first target clipped by crop boundary. No raw/cooked/contact ground truth.'))
  for box in local:ImageDraw.Draw(im).rectangle(box,outline='lime',width=2)
  im.save(E/f'{i}_overlay.jpg')
 paths=(B/'food_aligned_replay_v1/train.txt').read_text(encoding='utf-8').splitlines()+[r['image'] for r in records];assert len(paths)==657 and len({sha(p) for p in paths})==657
 (E/'train.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8');(E/'manifest.json').write_text(json.dumps(dict(records=records,base_manifest_sha256=sha(B/'food_aligned_replay_v1/manifest.json'),scope='4 correlated frames from1 historically viewed video, not 4 independent sources. Only training supplement, not held-out test.'),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
