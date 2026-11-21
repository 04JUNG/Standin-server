"""Bake authored anatomical paths into UV distance masks, using rest-surface samples.
R=centerline, G=construction, B=muscle. White means curve center; zero is >=6mm.
No camera-space decals, generated anatomy images, wireframe or noise.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image

P=argparse.ArgumentParser();P.add_argument('--assets',required=True);P.add_argument('--id',required=True);P.add_argument('--resolution',type=int,default=4096);a=P.parse_args();root=Path(a.assets);N=a.resolution
style=json.loads((Path(__file__).with_name('skin_style.json')).read_text())
z=np.load(root/(a.id+'-surface.npz'));pos=z['positions'];tris=z['triangles'];uvs=z['uv'];h=pos[:,2].max()/1.78
xyz=np.full((N,N,3),np.nan,np.float32);occupied=np.zeros((N,N),bool);overlaps=0;overlap_pixels=[]
for inds,uv in zip(tris,uvs):
 q=uv*np.array([N,-N])+np.array([0,N]);lo=np.maximum(np.floor(q.min(0)).astype(int),0);hi=np.minimum(np.ceil(q.max(0)).astype(int),N-1)
 if np.any(hi<lo):continue
 yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];x=xx+.5;y=yy+.5
 den=(q[1,1]-q[2,1])*(q[0,0]-q[2,0])+(q[2,0]-q[1,0])*(q[0,1]-q[2,1])
 if abs(den)<1e-10:continue
 b0=((q[1,1]-q[2,1])*(x-q[2,0])+(q[2,0]-q[1,0])*(y-q[2,1]))/den
 b1=((q[2,1]-q[0,1])*(x-q[2,0])+(q[0,0]-q[2,0])*(y-q[2,1]))/den;b2=1-b0-b1
 inside=(b0>=0)&(b1>=0)&(b2>=0)
 block=xyz[lo[1]:hi[1]+1,lo[0]:hi[0]+1];block[inside]=(b0[...,None]*pos[inds[0]]+b1[...,None]*pos[inds[1]]+b2[...,None]*pos[inds[2]])[inside]/h
 old=occupied[lo[1]:hi[1]+1,lo[0]:hi[0]+1];overlap=old&inside;overlaps+=int(overlap.sum());overlap_pixels.extend(((yy[overlap]*N+xx[overlap]).astype(int)).tolist());old|=inside
# Nearest edge padding for filtering: carry surface positions, never enlarge ink.
core=occupied.copy()
for _ in range(5):
 old=occupied.copy()
 for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
  neighbor=np.roll(old,(dy,dx),(0,1));take=(~occupied)&neighbor
  if dy==1:take[0,:]=False
  if dy==-1:take[-1,:]=False
  if dx==1:take[:,0]=False
  if dx==-1:take[:,-1]=False
  xyz[take]=np.roll(xyz,(dy,dx),(0,1))[take];occupied[take]=True
# Spatial buckets for front/back rays onto the actual rest mesh.
T=pos[tris]/h;grid={};cell=.025
for ti,triangle in enumerate(T):
 lo=np.floor(triangle[:,[0,2]].min(0)/cell).astype(int);hi=np.floor(triangle[:,[0,2]].max(0)/cell).astype(int)
 for ix in range(lo[0],hi[0]+1):
  for iz in range(lo[1],hi[1]+1):grid.setdefault((ix,iz),[]).append(ti)
def project_to_surface(points,side):
 result=[]
 for x,z in points:
  ids=grid.get((int(np.floor(x/cell)),int(np.floor(z/cell))),[])
  if not ids:result.append([np.nan]*3);continue
  t=T[ids];q=t[:,:,[0,2]]
  den=(q[:,1,1]-q[:,2,1])*(q[:,0,0]-q[:,2,0])+(q[:,2,0]-q[:,1,0])*(q[:,0,1]-q[:,2,1]);valid=np.abs(den)>1e-12;den=np.where(valid,den,1)
  b0=((q[:,1,1]-q[:,2,1])*(x-q[:,2,0])+(q[:,2,0]-q[:,1,0])*(z-q[:,2,1]))/den
  b1=((q[:,2,1]-q[:,0,1])*(x-q[:,2,0])+(q[:,0,0]-q[:,2,0])*(z-q[:,2,1]))/den;b2=1-b0-b1
  valid&=(b0>=-1e-5)&(b1>=-1e-5)&(b2>=-1e-5)
  ys=(b0*t[:,0,1]+b1*t[:,1,1]+b2*t[:,2,1])[valid]
  result.append([x,float(ys.min() if side=='front' else ys.max()),z] if len(ys) else [np.nan]*3)
 return np.array(result,dtype=np.float32)
print('rasterized',a.id,'pixels',int(occupied.sum()),'overlap',overlaps,flush=True)
samples=xyz[occupied];X,Y,Z=samples.T
masks=np.zeros((len(samples),3),np.float32);paths=[];radius=.006

def smooth(points):
 p=np.array(points,dtype=np.float32);p=np.vstack([p[0],p,p[-1]]);out=[]
 for i in range(1,len(p)-2):
  t=np.linspace(0,1,max(4,int(np.linalg.norm(p[i+1]-p[i])/.002)))[:,None]
  out.extend(.5*((2*p[i])+(-p[i-1]+p[i+1])*t+(2*p[i-1]-5*p[i]+4*p[i+1]-p[i+2])*t*t+(-p[i-1]+3*p[i]-3*p[i+1]+p[i+2])*t*t*t))
 return np.array(out,dtype=np.float32)

def path(name,points,channel=2,side='front',mirror=True,projected=False,z_min=None,z_max=None):
 if (channel==0 or projected) and side=='both':
  path(name,points,channel,'front',mirror,projected,z_min,z_max);path(name,points,channel,'back',mirror,projected,z_min,z_max);return
 for sign in ([1,-1] if mirror else [1]):
  pts=np.array(points,dtype=np.float32);pts[:,0]*=sign
  # Match rest bone stations; women have a shorter upper arm, longer forearm.
  if source.get('characterId') in ('male-base-v0-8-0','female-base-v0-4-0'):
   xx=np.abs(pts[:,0]);stations_x=[0,float(bones['LeftArm'][0]),float(bones['LeftForeArm'][0]),float(bones['LeftHand'][0]),.9]
   pts[:,0]=np.sign(pts[:,0])*np.interp(xx,[0,.165,.453,.701,.9],stations_x)
  elif a.id=='female':
   xx=np.abs(pts[:,0]);pts[:,0]=np.sign(pts[:,0])*np.interp(xx,[0,.165,.453,.701,.9],[0,.167,.420,.693,.9])
  s=smooth(pts)
  front=(Y<.025) if side=='front' else (Y>.025) if side=='back' else np.ones_like(Y,dtype=bool)
  select=front&(X>=s[:,0].min()-radius)&(X<=s[:,0].max()+radius)&(Z>=s[:,1].min()-radius)&(Z<=s[:,1].max()+radius)
  if channel in (0,2) or projected:
   s=project_to_surface(s,side);finite=np.isfinite(s[:,0])
   if finite.sum()<2:continue
   good=s[finite]
   select=(X>=good[:,0].min()-radius)&(X<=good[:,0].max()+radius)&(Z>=good[:,2].min()-radius)&(Z<=good[:,2].max()+radius)&(Y>=good[:,1].min()-radius)&(Y<=good[:,1].max()+radius)
  if z_min is not None:select&=Z>=z_min
  if z_max is not None:select&=Z<=z_max
  if name.startswith('shared torso-to-leg meridian'):select&=GZ<=collar_z+.0005
  idx=np.flatnonzero(select);v=samples[idx] if channel in (0,2) or projected else np.column_stack([X[idx],Z[idx]]);dist=np.full(len(idx),radius,np.float32)
  for p,q in zip(s,s[1:]):
   if not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q-p)>.035:continue
   d=q-p;t=np.clip(((v-p)*d).sum(1)/max(float(d@d),1e-12),0,1);dist=np.minimum(dist,np.linalg.norm(v-(p+t[:,None]*d),axis=1))
  masks[idx,channel]=np.maximum(masks[idx,channel],np.clip(1-dist/radius,0,1))
  paths.append({'name':name+(' L' if sign==1 else ' R') if mirror else name,'channel':channel,'surface':side,'points':pts.tolist(),'projected':projected or channel in (0,2),'z_min':z_min,'z_max':z_max})

# CSP Ver.2 line inventory: observed front + user-supplied profile/back.
# Cross sections are complete 3D surface intersections, including side faces.
spec=json.loads(Path(__file__).with_name('guide_spec.json').read_text())
source=json.loads((root/(a.id+'-source.json')).read_text())
bones={k.split(':')[-1]:np.array(v)/h for k,v in source['bone_heads'].items()}
# The same named stations follow each source rig's proportions, without changing it.
stations=np.array([0,.10036,.51381,.96232,1.10440,1.21095,1.33272,1.46971,1.56210,1.78])
actual=np.array([0,*[bones[n][2] for n in ['LeftFoot','LeftLeg','LeftUpLeg','Spine','Spine1','Spine2','Neck','Head']],1.78])
GZ=np.interp(Z,actual,stations)
def section(name,distance,select,channel=1):
 idx=np.flatnonzero(select&(np.abs(distance)<radius))
 masks[idx,channel]=np.maximum(masks[idx,channel],1-np.abs(distance[idx])/radius)
 paths.append({'name':name,'channel':channel,'surface':'closed surface section','projected':False,'construction':'implicit plane/surface intersection'})

section('sagittal center',X,GZ>=.88,0)
section('eye level',GZ-spec['head']['eye_z'],GZ>1.56)
head_y=np.interp(GZ,[1.46,1.56,1.78],[.050,.030,.030])
# Collar rises over the shoulder and nape, as seen in the profile reference.
collar_z=np.interp(Y,[-.10,.035,.13],[1.442,1.475,1.486])
section('head side meridian',Y-head_y,(GZ>=collar_z)&(np.abs(X)<.13))
section('neck base', (GZ-collar_z)/1.03,(np.abs(X)<.145)&(GZ>1.40)&(GZ<1.52))
for name,zz in spec['torso']['cross_sections'].items():
 section(name,GZ-zz,(np.abs(X)<.205)&(GZ>.95)&(GZ<1.37))
join=style['guide_join'];join_z=join['height_m_in_1_78m_rest_model'];overlap=join['overlap_m'];meridian=join['shared_curve']
for side in ['front','back']:
 # Female stations are remapped before surface projection; muscle authoring stays unchanged.
 pts=np.array(meridian);pts[:,1]=np.interp(pts[:,1],stations,actual)
 path('shared torso-to-leg meridian gray',pts,1,side,projected=True,z_min=np.interp(join_z-overlap,stations,actual))
 path('shared torso-to-leg meridian orange',pts,0,side,projected=True,z_max=np.interp(join_z+overlap,stations,actual))
for name,zz in spec['leg']['cross_sections'].items():
 section(name,GZ-zz,(np.abs(X)<.22)&(GZ<.95))
leg_y=np.interp(GZ,[.10,.514,.96],[bones['LeftFoot'][1],bones['LeftLeg'][1],bones['LeftUpLeg'][1]])
section('leg side meridians',Y-leg_y,(GZ<join_z)&(GZ>0)&(np.abs(X)<.22))
for sign,side in [(1,'Left'),(-1,'Right')]:
 arm,elbow,hand=[bones[side+n] for n in ['Arm','ForeArm','Hand']]
 ax=sign*X
 arm_y=np.interp(ax,[.09,sign*arm[0],sign*elbow[0],sign*hand[0],.79],[.045,arm[1],elbow[1],hand[1],.045])
 arm_z=np.interp(ax,[sign*arm[0],sign*elbow[0],sign*hand[0]],[arm[2],elbow[2],hand[2]])
 region=(ax>.09)&(ax<.79)&(Z>arm_z-.09)&(Z<arm_z+.12)&((ax>.19)|(GZ>1.438))
 # Top/bottom arm meridians are orange; front/back meridians are gray.
 section('arm superior/inferior orange '+side,Y-arm_y-.012,region,0)
 section('arm anterior/posterior gray '+side,Z-arm_z,region&(ax>sign*arm[0]))
 for name,t in spec['arm']['cross_section_fractions'].items():
  xx=np.interp(t,[0,1,2],[sign*arm[0],sign*elbow[0],sign*hand[0]])
  section(name+' '+side,ax-xx,region)
 # A gray side meridian reaches the heel; two transverse foot sections divide instep/toes.
 for name,yy in spec['foot']['cross_sections_y'].items():
  section(name+' '+side,Y-yy,(ax>.015)&(ax<.18)&(GZ<.16))

# Latest V2 authored path intent, rebound to current neutral surface.
# Old-model triangle/barycentric IDs are never transferred.
for entry in json.loads(Path(__file__).with_name(a.id+'-muscle-paths.v2.json').read_text()):
 pts=np.asarray(entry['points'],dtype=np.float32)
 pts[:,1]=np.interp(pts[:,1],stations,actual)
 path(entry['name'],pts,2,entry['surface'],False)
# Front/top foot meridian continues the leg axis to the toe box.
foot=(Z<.11)&(Z>.006)
for sign in [1,-1]:
 desired=sign*(.073+.09*(Y-.04))
 d=np.abs(X-desired);idx=np.flatnonzero(foot&(d<radius))
 masks[idx,0]=np.maximum(masks[idx,0],1-d[idx]/radius)
mask=np.zeros((N,N,3),np.uint8);mask[occupied]=(masks*255+.5).astype(np.uint8)
# Gutter colors were evaluated from padded rest positions above.
Image.fromarray(mask).save(root/(a.id+'-masks.png'))
(root/(a.id+'-paths.json')).write_text(json.dumps(paths,indent=2))
(root/(a.id+'-uv-report.json')).write_text(json.dumps({'resolution':N,'covered_pixels':int(core.sum()),'uv_overlap_pixels':overlaps,'uv_overlap_fraction':overlaps/max(1,int(core.sum())),'gutter_px':5,'distance_range_m':radius,'authored_paths':len(paths),'overlap_pixels_near_ink':int((mask.reshape(-1,3)[overlap_pixels].max(axis=1)>127).sum()) if overlap_pixels else 0},indent=2))
# Portable ordinary color textures; the preview can recompose these from masks.
linear=np.array([.72,.74,.76]);gray=np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055)*255
for name,base,guide,muscle in [('gray-guide',gray,True,False),('flesh',style['flesh']['reference_sample_rgb'],False,False),('flesh-guide',style['flesh']['reference_sample_rgb'],True,False),('muscle',gray,False,True)]:
 color=np.empty((N,N,3),np.float32);color[:]=base
 def ink(ch,col,width,opacity):
  v=mask[:,:,ch].astype(np.float32)/255;coverage=np.clip((v-(1-width/radius))/.13+.5,0,1)*opacity
  color[:]=color*(1-coverage[:,:,None])+np.array(col)*coverage[:,:,None]
 if guide:
  for ch,key in [(1,'gray'),(0,'orange')]:
   line=style['lines'][key];ink(ch,line['rgb'],line['half_width_m'],line['opacity'])
 if muscle:ink(2,[133,139,145],.0008,.78)
 Image.fromarray(np.clip(color,0,255).astype(np.uint8)).save(root/(a.id+'-'+name+'.png'))
print('TEXTURES READY',a.id,flush=True)
