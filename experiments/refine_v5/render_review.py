"""Static review of actual saved native meshes; no image generation service."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from scipy.spatial.transform import Rotation
from experiments.refine_v5.learning_data import frame

def canonical(mesh,bones):
    _,c,s,f=frame(bones);return (np.array(mesh['vertices'])-c)@f/s,np.array(mesh['faces'],int)
def render(vertices,faces,size=(260,330),yaw=0,framing=None):
    v=vertices@Rotation.from_euler('y',yaw,degrees=True).as_matrix().T
    xy=v[:,:2]*np.array([1,-1]);scale,origin=framing;xy=xy*scale+origin
    return render_projected(v, faces, xy, size)

def render_projected(vertices, faces, xy, size):
    image = np.full((size[1], size[0], 3), 248, np.uint8)
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    light = np.array([-.3, .5, -.8])
    shade = np.clip(abs(normal @ light), 0, 1) * 75 + 145
    pixels = xy.astype('int32')
    for i in np.argsort(triangles[:, :, 2].mean(1)):
        color = int(shade[i])
        cv2.fillConvexPoly(image, pixels[faces[i]], (color, color + 8, color + 12))
    return Image.fromarray(image)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--previous',required=True);a=p.parse_args();run=Path(a.run);old=Path(a.previous)
    preview=json.loads((run/'native-audit/preview-meshes.json').read_text());gt={r['case_id']:r for r in json.loads((old/'data/native_audit.json').read_text())};rigs={r['case_id']:r for r in json.loads((run/'native-data/records.json').read_text()) if r['status']=='ready'}
    cases=sorted({key.split('.')[0] for key in preview if '.candidate.r1' in key});cellw,cellh=260,370
    out=Image.new('RGB',(cellw*5,55+cellh*len(cases)),'white');draw=ImageDraw.Draw(out)
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',17)
    except OSError:font=ImageFont.load_default()
    headers=['Rough','B0 body-frame front','Saved v5 front (diagnostic)','GT body-frame front','Saved v5 35 degree']
    for col,h in enumerate(headers):draw.text((col*cellw+8,8),h,fill='black',font=font)
    for row,case in enumerate(cases):
        rough=Image.open(next((old/'data/러프').glob(case+'.*'))).convert('RGB');rough.thumbnail((250,330));y=55+row*cellh
        out.paste(rough,((cellw-rough.width)//2,y+25));draw.text((8,y),case,fill='black',font=font)
        base=preview[case+'.base'];final=preview[case+'.candidate.r1'];mesh=json.loads(Path(rigs[case]['gt_mesh']).read_text())
        jobs=[]
        for col,m,bones,yaw in [(1,base,base['bones'],0),(2,final,final['bones'],0),(3,mesh,gt[case]['bones'],0),(4,final,final['bones'],35)]:
            v,f=canonical(m,bones);jobs.append((col,v,f,yaw))
        allxy=np.concatenate([(v@Rotation.from_euler('y',yaw,degrees=True).as_matrix().T)[:,:2]*np.array([1,-1]) for _,v,_,yaw in jobs]);lo=allxy.min(0);hi=allxy.max(0)
        scale=min(240/max(hi[0]-lo[0],1e-5),310/max(hi[1]-lo[1],1e-5));origin=np.array([130,165])-(lo+hi)/2*scale
        for col,v,f,yaw in jobs:out.paste(render(v,f,yaw=yaw,framing=(scale,origin)),(col*cellw,y+25))
    draw.text((8,out.height-18),'Body-frame review; GT rough camera unavailable. Diagnostic files differ from safe B0 fallback.',fill='black')
    out.save(run/'native-review.png')
if __name__=='__main__':main()
