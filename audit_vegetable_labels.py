"""Audit train/validation only; do not mine the existing test set for training fixes."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

BASE=Path(__file__).resolve().parent
RAW=BASE.parent/'work'/'datasets'

def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=x*y;return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area+1e-12)

def main():
    names=dict(csv.reader((RAW/'oi_classes.csv').open(encoding='utf-8')))
    selected_names={'Broccoli','Carrot','Cucumber','Cabbage','Potato','Bell pepper','Asparagus','Zucchini','Radish','Artichoke'}
    children={k:v for k,v in names.items() if v in selected_names}
    manifest=json.loads((BASE/'vegetable_dataset'/'manifest.json').read_text())
    records=[r for r in manifest['images'] if r['split'] in {'train','validation'}]
    all_results={};examples=[]
    for split in ['train','validation']:
        ids={r['image_id'] for r in records if r['split']==split};parents={};child_rows={i:[] for i in ids}
        invalid=[];small=0;total=0
        for image_id in ids:
            boxes=[]
            for line in (BASE/'vegetable_dataset'/'labels'/split/f'{image_id}.txt').read_text().splitlines():
                cls,x,y,w,h=map(float,line.split());total+=1
                box=[x-w/2,y-h/2,x+w/2,y+h/2]
                if w<=0 or h<=0 or min(box)<-1e-6 or max(box)>1+1e-6:invalid.append(image_id)
                if min(w,h)*640<16:small+=1
                boxes.append(box)
            parents[image_id]=boxes
        with (RAW/f'oi_{split}.csv').open() as f:
            header=next(csv.reader([next(f)]))
            for line in f:
                image_id=line.split(',',1)[0]
                if image_id not in ids:continue
                r=dict(zip(header,next(csv.reader([line]))))
                if r['LabelName'] in children and r['IsGroupOf']=='0' and r['IsDepiction']=='0':
                    box=[float(r[k]) for k in ['XMin','YMin','XMax','YMax']]
                    child_rows[image_id].append({'name':children[r['LabelName']],'box':box,
                        'best_parent_iou':max([iou(box,p) for p in parents[image_id]] or [0])})
        absent=[{'image_id':i,'split':split,**r} for i,rs in child_rows.items() for r in rs if r['best_parent_iou']<.5]
        all_results[split]={'images':len(ids),'boxes':total,'invalid_boxes':invalid,
            'boxes_with_normalized_short_side_under_16_at_640':small,
            'selected_subclasses':children,'subclass_boxes':sum(map(len,child_rows.values())),
            'subclass_boxes_without_parent_iou_0_5':absent}
        examples.extend(absent)
    (BASE/'vegetable_label_audit.json').write_text(json.dumps(all_results,indent=2))
    # Visual screen of candidates, not automatic corrections.
    chosen=[];seen=set()
    for r in examples:
        if r['image_id'] not in seen:chosen.append(r);seen.add(r['image_id'])
        if len(chosen)>=12:break
    preview=Image.new('RGB',(960,720),'white')
    for i,r in enumerate(chosen):
        with Image.open(BASE/'vegetable_dataset'/'images'/r['split']/f"{r['image_id']}.jpg") as im:image=im.convert('RGB')
        w,h=image.size;d=ImageDraw.Draw(image)
        for line in (BASE/'vegetable_dataset'/'labels'/r['split']/f"{r['image_id']}.txt").read_text().splitlines():
            _,x,y,bw,bh=map(float,line.split());d.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline='red',width=3)
        x1,y1,x2,y2=r['box'];d.rectangle((x1*w,y1*h,x2*w,y2*h),outline='lime',width=4)
        image.thumbnail((320,155));px=i%3*320;py=i//3*180;preview.paste(image,(px,py));ImageDraw.Draw(preview).text((px+3,py+156),r['name']+' '+r['image_id'],fill='black')
    preview.save(BASE/'vegetable_audit_preview.jpg')
    print(json.dumps({s:{k:v for k,v in r.items() if k not in {'selected_subclasses','subclass_boxes_without_parent_iou_0_5'}}|{'unmatched_subclass_boxes':len(r['subclass_boxes_without_parent_iou_0_5'])} for s,r in all_results.items()},indent=2))

if __name__=='__main__':main()
