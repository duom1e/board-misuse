"""Post-training validation diagnostics only; never opens held-out test when gate fails."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
from run_experiment import box_iou,write_json
from experiment_data import read_boxes
from ultralytics import YOLO
BASE=Path(__file__).resolve().parent;RUN=BASE/'experiments/visor_adaptation_broad'

def main():
    c=json.loads((RUN/'comparison.json').read_text(encoding='utf-8'));audit=json.loads((RUN/'data_audit.json').read_text(encoding='utf-8'))
    records=[r for r in audit['records'] if r['split']=='val'];details={}
    for key,weights in c['weights'].items():
        model=YOLO(weights);rows=[]
        sheet=Image.new('RGB',(1280,((len(records)+3)//4)*240),'white')
        for i,r in enumerate(records):
            pred=model.predict(r['image'],conf=.5,imgsz=640,device=0,verbose=False)[0];h,w=pred.orig_shape
            gt=[[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h] for _,x,y,bw,bh in read_boxes(Path(r['label']),['cutting_board'])]
            used=set();detections=[]
            for box,score in sorted(zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist()),key=lambda p:-p[1]):
                overlap,j=max(((box_iou(box,g),j) for j,g in enumerate(gt) if j not in used),default=(0,None));ok=overlap>=.5 and j is not None
                if ok:used.add(j)
                detections.append({'box':box,'confidence':score,'matched':ok,'best_available_iou':overlap})
            row={'image':r['image'],'gt':gt,'predictions':detections,'tp':len(used),'fp':len(detections)-len(used),'fn':len(gt)-len(used)};rows.append(row)
            im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
            for g in gt:draw.rectangle(g,outline='orange',width=5)
            for p in detections:draw.rectangle(p['box'],outline='cyan',width=4)
            im.thumbnail((315,205));x=i%4*320;y=i//4*240;sheet.paste(im,(x,y));draw=ImageDraw.Draw(sheet);draw.text((x,y+209),Path(r['image']).stem,fill='black');draw.text((x,y+224),f'TP {row["tp"]} FP {row["fp"]} FN {row["fn"]}',fill='black')
        counts={k:sum(r[k] for r in rows) for k in ['tp','fp','fn']}
        if counts!={k:c['validation'][key]['micro'][k] for k in counts}:raise RuntimeError('Diagnostic count mismatch; investigate')
        details[key]={'counts_match_formal_evaluation':True,'rows':rows};sheet.save(RUN/(key+'_validation_predictions.jpg'))
    write_json(RUN/'validation_diagnostics.json',details)
    print({key:entry['counts_match_formal_evaluation'] for key,entry in details.items()})

if __name__=='__main__':main()
