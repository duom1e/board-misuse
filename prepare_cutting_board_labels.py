"""Manually reviewed single-source board boxes, excluding uncertain food labels."""
import hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent/'kitchen_cutting_batch_v1'
BOXES={330:[120,0,406,171],360:[125,0,414,212],450:[250,20,505,255],510:[180,0,453,202],540:[265,34,525,266],570:[179,0,464,220]}

def main():
    e=B/'board_labels_v1';e.mkdir(exist_ok=True);rows=[];sheet=Image.new('RGB',(1280,1140))
    results=json.loads((B/'live/results.json').read_text(encoding='utf-8'))['rows']
    for i,(t,box) in enumerate(BOXES.items()):
        p=B/f'{t}.jpg';im=Image.open(p).convert('RGB');w,h=im.size;a,b,c,d=box
        assert 0<=a<c<=w and 0<=b<d<=h
        (e/f'{t}.txt').write_text(f'0 {(a+c)/2/w:.8f} {(b+d)/2/h:.8f} {(c-a)/w:.8f} {(d-b)/h:.8f}\n',encoding='utf-8')
        pred=next(r for r in results if r['requested_seconds']==t)
        boxes=pred['output_validation']['normalized_boxes']['boards'];ious=[]
        for q in boxes:
            x,y,z,v=[q[0]*w/1000,q[1]*h/1000,q[2]*w/1000,q[3]*h/1000]
            inter=max(0,min(c,z)-max(a,x))*max(0,min(d,v)-max(b,y));ious.append(inter/((c-a)*(d-b)+(z-x)*(v-y)-inter))
        rows.append(dict(seconds=t,image=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),box_xyxy=box,best_iou=max(ious,default=0),top_truncated=b==0,review='assistant_manual_not_expert',source_group='BV1dHckzwEBA',split='development_consumed',training_eligible=False))
        ImageDraw.Draw(im).rectangle(box,outline='lime',width=3);sheet.paste(im,(i%2*640,i//2*380));ImageDraw.Draw(sheet).text((i%2*640,i//2*380+362),str(t),fill='white')
    sheet.save(e/'overlay.jpg');(e/'manifest.json').write_text(json.dumps(dict(records=rows,license='unverified',scope='Six selected visible-board frames, not exhaustive video labels or independent test. Coordinates manually set after full-size image inspection; food identity unknown. No training.'),indent=2),encoding='utf-8')
    print([(r['seconds'],round(r['best_iou'],3)) for r in rows])
if __name__=='__main__':main()
