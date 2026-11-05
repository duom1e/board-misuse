from pathlib import Path
import json,hashlib,shutil
B=Path(__file__).resolve().parent
if __name__=='__main__':
 O=B/'experiments/board_multisource_bbox_v1';O.mkdir(exist_ok=False)
 old=json.loads((B/'experiments/board_manual_v2/manifest.json').read_text())['records']
 proposed=json.loads((B/'experiments/board_boundary_revision_v2/all_five_reviewed.json').read_text())
 rows={r['sha256']:{'image':r['original'],'box':r['box_px'],'sha256':r['sha256']} for r in old}
 for r in proposed:
  assert r['sha256']==hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()
  rows[r['sha256']]={'image':r['image'],'box':r['proposed_box_px'],'sha256':r['sha256']}
 from PIL import Image
 manifest=[];train=(B/'board_dataset/train.txt').read_text().splitlines();val=[]
 for r in rows.values():
  p=Path(r['image']);source=p.name.split('_')[0];split='val' if source=='BV1pu411U7Zh' else 'train'
  with Image.open(p) as im:w,h=im.size
  x1,y1,x2,y2=r['box'];assert 0<=x1<x2<=w and 0<=y1<y2<=h
  dst=O/'images'/split/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
  lab=O/'labels'/split/(p.stem+'.txt');lab.parent.mkdir(parents=True,exist_ok=True);lab.write_text(f'0 {(x1+x2)/2/w} {(y1+y2)/2/h} {(x2-x1)/w} {(y2-y1)/h}\n')
  (train if split=='train' else val).extend([str(dst)]*(3 if split=='train' else 1))
  manifest.append({**r,'source':source,'split':split,'target':str(dst),'label_scope':'visible board object bbox; not contact surface','review':'single assistant approximate; original and overlays reviewed'})
 assert not {r['source'] for r in manifest if r['split']=='train'}&{r['source'] for r in manifest if r['split']=='val'}
 assert not {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in train}&{hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in val}
 for name,items in [('train',train),('val',val)]: (O/(name+'.txt')).write_text('\n'.join(items)+'\n')
 (O/'data.yaml').write_text(f'path: {O.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: cutting_board\n')
 (O/'manifest.json').write_text(json.dumps({'records':manifest,'train_entries':len(train),'val':len(val),'original_data_preserved':True},ensure_ascii=False,indent=2),encoding='utf-8')
 print({'new_train':sum(r['split']=='train' for r in manifest),'val':len(val),'train_entries':len(train)})
