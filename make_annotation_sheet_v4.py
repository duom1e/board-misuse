"""给下一批待标帧生成带网格的人工查看表。"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE=Path(__file__).resolve().parent
OUT=BASE/'experiments'/'board_box_preview_v4'
ITEMS=[
 ('new_source_video_v3','BV1pq4y147jk_080s.jpg'),
 ('new_source_video_v3','BV1pq4y147jk_140s.jpg'),
 ('new_source_video_v3','BV1pq4y147jk_260s.jpg'),
 ('new_source_video_v3','BV1pq4y147jk_380s.jpg'),
 ('new_source_video_v4','BV1bqsfzbELh_020s.jpg'),
 ('new_source_video_v4','BV1bqsfzbELh_090s.jpg'),
 ('new_source_video_v4','BV1bqsfzbELh_335s.jpg'),
 ('new_source_video_v4','BV1bqsfzbELh_405s.jpg'),
]
W,H=480,270

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',13)
    canvas=Image.new('RGB',(W*2,H*4),'white')
    for i,(folder,name) in enumerate(ITEMS):
        im=Image.open(BASE/'experiments'/folder/'frames'/name).convert('RGB').resize((W,H))
        d=ImageDraw.Draw(im)
        for k in range(1,10):
            x=W*k/10; y=H*k/10
            d.line((x,0,x,H),fill='red',width=1);d.line((0,y,W,y),fill='red',width=1)
            d.text((x+1,1),str(k*10),fill='yellow',font=font);d.text((1,y+1),str(k*10),fill='yellow',font=font)
        canvas.paste(im,((i%2)*W,(i//2)*H))
        ImageDraw.Draw(canvas).text(((i%2)*W+3,(i//2)*H+H-16),f'#{i} {name}',fill='lime',font=font)
    canvas.save(OUT/'grid.jpg',quality=90)
    (OUT/'items.json').write_text(json.dumps([{'folder':f,'frame':n} for f,n in ITEMS],ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT/'grid.jpg')

if __name__=='__main__':main()
