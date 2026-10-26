"""Validation error breakdown for the preplanned capacity candidate; no test use."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
from run_experiment import box_iou,write_json
from experiment_data import read_boxes
from ultralytics import YOLO
BASE=Path(__file__).resolve().parent
RUN=BASE/'experiments/board_expanded_s_reviewed'

def main():
    audit=json.loads((RUN/'data_audit.json').read_text(encoding='utf-8'))
    comparison=json.loads((RUN/'comparison.json').read_text(encoding='utf-8'))
    output={}
    for key,entry in comparison['results'].items():
        model=YOLO(entry['weights']);rows=[];fp=[]
        for record in audit['records']:
            if record['split']!='val':continue
            prediction=model.predict(record['image'],conf=.5,imgsz=640,device=0,verbose=False)[0]
            h,w=prediction.orig_shape
            gt=[[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h] for _,x,y,bw,bh in read_boxes(Path(record['label']),['cutting_board'])]
            used=set();matched={}
            detections=sorted(zip(prediction.boxes.xyxy.cpu().tolist(),prediction.boxes.conf.cpu().tolist()),key=lambda x:-x[1])
            for box,conf in detections:
                score,index=max(((box_iou(box,g),i) for i,g in enumerate(gt) if i not in used),default=(0,None))
                if score>=.5 and index is not None:used.add(index);matched[index]=(box,conf,score)
                else:fp.append({'image':record['image'],'box':box,'confidence':conf})
            for i,g in enumerate(gt):
                short=min(g[2]-g[0],g[3]-g[1])*640/max(w,h)
                rows.append({'image':record['image'],'gt_box':g,'short_side_at640':short,
                             'bucket':'below64' if short<64 else '64to128' if short<128 else 'atleast128',
                             'matched':i in used,'prediction':matched.get(i)})
        groups={}
        for bucket in ['below64','64to128','atleast128']:
            subset=[r for r in rows if r['bucket']==bucket];n=sum(r['matched'] for r in subset)
            groups[bucket]={'gt':len(subset),'matched':n,'recall':n/len(subset) if subset else None}
        output[key]={'size_groups':groups,'ground_truth':rows,'false_predictions':fp}
        missed=[r for r in rows if not r['matched']]
        sheet=Image.new('RGB',(1280,max(1,(len(missed)+3)//4)*250),'#eeeeee')
        for i,row in enumerate(missed):
            with Image.open(row['image']) as original:
                im=original.convert('RGB');d=ImageDraw.Draw(im);d.rectangle(row['gt_box'],outline='orange',width=4)
                im.thumbnail((310,215))
            x=i%4*320;y=i//4*250;sheet.paste(im,(x,y));d=ImageDraw.Draw(sheet)
            d.text((x+2,y+218),Path(row['image']).stem,fill='black')
            d.text((x+2,y+233),f'short640={row["short_side_at640"]:.0f}',fill='black')
        sheet.save(RUN/(key+'_misses.jpg'))
    write_json(RUN/'error_breakdown.json',output)
    print(json.dumps({k:v['size_groups'] for k,v in output.items()}))

if __name__=='__main__':main()
