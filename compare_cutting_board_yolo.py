import json,hashlib,os
from pathlib import Path
from PIL import Image,ImageDraw
os.environ['YOLO_OFFLINE']='True'
os.environ['YOLO_AUTOINSTALL']='False'
B=Path(__file__).resolve().parent;E=B/'kitchen_cutting_batch_v1'

def main():
    from ultralytics import YOLO
    original=json.loads((E/'manual_presence.json').read_text(encoding='utf-8'))
    revised=json.loads(json.dumps(original))
    for r in revised['records']:
        if r['seconds'] in (420,600):r['board_presence']='no_visible_board';r['revision_reason']='Full-size original review; contact sheet annotation mistaken. After model results, not blinded.'
    revised['scope']='Revised non-blinded diagnostic: 8 visible, 2 no visible board, 1 unknown; no physical absence/compliance claim.'
    (E/'manual_presence_v2.json').write_text(json.dumps(revised,ensure_ascii=False,indent=2),encoding='utf-8')
    cfg=json.loads((B/'active_models.json').read_text(encoding='utf-8'))['board'];model=YOLO(cfg['weights'])
    live=json.loads((E/'live/results.json').read_text(encoding='utf-8'))['rows'];rows=[]
    sheet=Image.new('RGB',(1280,2280))
    for i,r in enumerate(revised['records']):
        p=model.predict(r['image'],imgsz=640,rect=False,conf=.25,verbose=False)[0]
        boxes=[dict(box=q.xyxy[0].tolist(),confidence=float(q.conf[0])) for q in p.boxes]
        v=next(x for x in live if x['requested_seconds']==r['seconds'])
        rows.append(dict(seconds=r['seconds'],expected=r['board_presence'],yolo_present=bool(boxes),yolo_boxes=boxes,vlm_state=v['board_state']))
        im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im)
        for q in boxes:d.rectangle(q['box'],outline='orange',width=3)
        x=i%2*640;y=i//2*380;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x,y+362),f"{r['seconds']} expected={r['board_presence']} YOLO={bool(boxes)} VLM={v['board_state']}",fill='white')
    report=dict(rows=rows,weights=cfg['weights'],weights_sha256=hashlib.sha256(Path(cfg['weights']).read_bytes()).hexdigest(),conf=.25,imgsz=640,rect=False,training_performed=False,adopted=False)
    for name in ('yolo','vlm'):
        counts=dict(tp=0,fn=0,fp=0,tn=0,excluded=0)
        for r in rows:
            if r['expected']=='unknown':counts['excluded']+=1;continue
            yes=r['yolo_present'] if name=='yolo' else r['vlm_state']=='candidate_present'
            counts['tp' if yes and r['expected']=='visible' else 'fp' if yes else 'fn' if r['expected']=='visible' else 'tn']+=1
        report[name]=counts
    (E/'yolo_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');sheet.save(E/'yolo_review.jpg');print({k:report[k] for k in ('yolo','vlm')})
if __name__=='__main__':main()
