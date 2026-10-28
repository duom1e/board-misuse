"""Uniform source screening, preserving frame indices and source provenance."""
import argparse,hashlib,json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent.parent
def main():
    p=argparse.ArgumentParser();p.add_argument('ids',nargs='+');p.add_argument('--output',required=True);a=p.parse_args();e=Path(a.output);e.mkdir(exist_ok=True);records=[]
    for id in a.ids:
        video=B/'待导入视频'/f'kitchennew_{id}.mp4';meta=video.with_suffix('.info.json');info=json.loads(meta.read_text(encoding='utf-8'))
        c=cv2.VideoCapture(str(video));fps=c.get(5);count=int(c.get(7));assert fps>0 and count>0
        sheet=Image.new('RGB',(1280,960));draw=ImageDraw.Draw(sheet);frames=[]
        for i in range(12):
            idx=int((count-1)*(i+.5)/12);c.set(1,idx);ok,frame=c.read();assert ok
            path=e/f'{id}_{i:02}.jpg';assert not path.exists(),'Do not overwrite screened frames';assert cv2.imwrite(str(path),frame)
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((420,210));x=i%3*425;y=i//3*240;sheet.paste(im,(x,y));draw.text((x,y+215),f'{i:02} {idx/fps:.2f}s',fill='white')
            frames.append(dict(index=i,frame_index=idx,seconds=idx/fps,image=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        c.release();sheet.save(e/f'{id}_review.jpg')
        records.append(dict(id=id,title=info.get('title'),uploader=info.get('uploader'),url=info.get('webpage_url'),video=str(video),sha256=hashlib.sha256(video.read_bytes()).hexdigest(),duration=count/fps,width=frame.shape[1],height=frame.shape[0],frames=frames,license='unverified',usage='screening_consumed_not_blind_test'))
    (e/'sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8');print([(r['id'],r['duration']) for r in records])
if __name__=='__main__':main()
