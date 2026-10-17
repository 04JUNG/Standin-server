"""Scientific raster alignment panels, fixed first three development images."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw


def preview(folder):
    rows=json.loads((folder/'rankings.json').read_text())[:3]
    canvas=Image.new('RGB',(768,3*290),'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        with np.load(folder/'masks'/(Path(row['image']).stem+'.npz'),allow_pickle=False) as data:
            query=data['query'];base=1
            selected=row['shortlist'].index(row['contour100'][1])
            for col,k in enumerate((None,base,selected)):
                image=np.full((256,256,3),255,np.uint8)
                image[query]=(45,45,45)
                if k is not None:
                    candidate=data[f'candidate_{k}']
                    image[candidate]=(220,70,70);image[candidate&query]=(50,60,190)
                x,y=col*256,i*290
                canvas.paste(Image.fromarray(image),(x,y+28))
                label='query' if k is None else ('baseline #2' if col==1 else 'contour #2')
                draw.text((x+4,y+3),row['image']+' '+label,fill='black')
    canvas.save(folder/'alignment-preview.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    preview(parser.parse_args().out)
