"""Render saved review samples; no model inference or changes to raw results."""
from pathlib import Path
import json,math
def render_contact_sheets(out,rows):
    from PIL import Image,ImageDraw
    pages=[]
    for start in range(0,len(rows),16):
        subset=rows[start:start+16];sheet=Image.new('RGB',(1200,math.ceil(len(subset)/4)*220),(25,25,25));draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(subset):
            x=i%4*300;y=i//4*220;path=r.get('review_image') or r.get('image')
            if path and Path(path).exists():
                with Image.open(path) as source:
                    im=source.convert('RGB');im.thumbnail((290,175));sheet.paste(im,(x+5,y+5))
            if r['board_state']=='unknown':draw.rectangle((x+1,y+1,x+298,y+218),outline='red',width=3)
            draw.text((x+5,y+183),f"{r['frame_seconds']:.3f}s  frame {r['frame_index']}",fill='white')
            draw.text((x+5,y+200),r['board_state'],fill='white')
        page=out/f'review_page_{start//16+1:02}.jpg';sheet.save(page);pages.append(page)
    return pages
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output_dir',type=Path);args=p.parse_args()
    data=json.loads((args.output_dir/'results.json').read_text(encoding='utf-8'))
    for path in render_contact_sheets(args.output_dir,data['rows']):print(path)
