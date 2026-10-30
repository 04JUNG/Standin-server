"""Assemble actual model renders into labelled inspection sheets."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ap=argparse.ArgumentParser()
ap.add_argument('--candidate',type=Path,required=True)
args=ap.parse_args()
out=args.candidate.parent
images=args.candidate/'images'
font=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',19)

def sheet(name,columns,views,width=350):
    height=round(width*820/700)
    canvas=Image.new('RGB',(len(columns)*width,len(views)*(height+34)),(20,24,31))
    draw=ImageDraw.Draw(canvas)
    for col,(label,stem) in enumerate(columns):
        for row,(view,view_label) in enumerate(views):
            image=Image.open(images/f'{stem}-{view}.png').convert('RGB')
            image=image.resize((width,height),Image.Resampling.LANCZOS)
            y=row*(height+34)
            canvas.paste(image,(col*width,y+34))
            draw.text((col*width+10,y+7),label+' · '+view_label,font=font,fill=(235,240,248))
    canvas.save(out/name)

views=[('front','정면'),('45','45도'),('side','측면')]
sheet('comparison.png',[
    ('기존 최소 -1','old-T-0'),('새 최소 -1','new-T-0'),('기본 0','new-T-4'),
    ('기존 최대 +1','old-T-8'),('확장 최대 +2','new-T-12')],views)
sheet('size-path.png',[(label,f'new-T-{i}') for i,label in [(4,'기본 0'),(3,'축소 25%'),(2,'축소 50%'),
    (1,'축소 75%'),(0,'축소 100%'),(8,'기존 최대 +1'),(10,'확장 중간 +1.5'),(12,'확장 최대 +2')]],views,280)
sheet('connections.png',[('새 최소 -1','detail-0'),('확장 최대 +2','detail-12')],views,500)
# One row per pose, columns at small/base/old maximum/extended maximum.
columns=[(label,str(i)) for i,label in [(0,'새 최소 -1'),(4,'기본 0'),(8,'기존 최대 +1'),(12,'확장 최대 +2')]]
height=410;width=350
canvas=Image.new('RGB',(1400,3*444),(20,24,31));draw=ImageDraw.Draw(canvas)
for row,(pose,label) in enumerate([('ARMS_DOWN','팔 내림'),('RAISE','팔 올림'),('BEND','몸 숙임')]):
    for col,(size,i) in enumerate(columns):
        im=Image.open(images/f'new-{pose}-{i}-45.png').resize((width,height),Image.Resampling.LANCZOS)
        canvas.paste(im,(col*width,row*444+34));draw.text((col*width+8,row*444+7),f'{label} · {size}',font=font,fill=(235,240,248))
canvas.save(out/'poses.png')
print('BREAST_SHEETS_COMPLETE',out)
