from pathlib import Path
import json,hashlib,shutil
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent
O=B/'experiments/board_manual_v2'
if (O/'manifest.json').exists():
    raise FileExistsError('Preserve existing annotations; use a new version')
# Image-coordinate enclosing rectangles of visible board object, including thickness.
labels=[(3,'BV12a41187Ja_090s.jpg',[0,0,1055,720],'train'),(3,'BV12a41187Ja_190s.jpg',[250,418,1075,720],'train'),(3,'BV12a41187Ja_240s.jpg',[345,420,1150,720],'train'),(5,'BV1pu411U7Zh_115s.jpg',[466,187,1280,720],'val'),(5,'BV1pu411U7Zh_140s.jpg',[400,173,1280,720],'val'),(5,'BV1pu411U7Zh_165s.jpg',[188,0,1225,684],'val')]
records=[]; canvas=Image.new('RGB',(960,900),'white')
for i,(v,name,box,split) in enumerate(labels):
 p=B/f'experiments/new_source_video_v{v}/frames'/name
 im=Image.open(p).convert('RGB');w,h=im.size
 dst=O/'images'/split/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
 lab=O/'labels'/split/(p.stem+'.txt');lab.parent.mkdir(parents=True,exist_ok=True)
 x1,y1,x2,y2=box;lab.write_text(f'0 {(x1+x2)/2/w} {(y1+y2)/2/h} {(x2-x1)/w} {(y2-y1)/h}\n')
 records.append({'image':str(dst),'original':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'box_px':box,'split':split,'source':name.split('_')[0],'review':'assistant visual estimate, nonexpert, occlusion/truncation uncertain'})
 ImageDraw.Draw(im).rectangle(box,outline='lime',width=5);im.thumbnail((480,270));x=i%2*480;y=i//2*300;canvas.paste(im,(x,y+25));ImageDraw.Draw(canvas).text((x+2,y+4),name+' '+split,fill='black')
canvas.save(O/'annotation_preview.jpg')
(O/'manifest.json').write_text(json.dumps({'records':records,'scope':'visible board object bounding rectangle, not surface mask','limitations':'Previously viewed development sources; source split only for this experiment. Approximate single-assistant labels.'},ensure_ascii=False,indent=2),encoding='utf-8')
base=(B/'board_dataset/train.txt').read_text().splitlines()
train=base+[r['image'] for r in records if r['split']=='train']*5
val=[r['image'] for r in records if r['split']=='val']
assert not set(r['sha256'] for r in records if r['split']=='train') & set(r['sha256'] for r in records if r['split']=='val')
(O/'train.txt').write_text('\n'.join(train)+'\n');(O/'val.txt').write_text('\n'.join(val)+'\n')
(O/'data.yaml').write_text(f'path: {O.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: cutting_board\n')
print({'train_entries':len(train),'new_train':3,'val':3})
