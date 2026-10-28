"""Bake authored anatomical paths into UV distance masks, using rest-surface samples.
R=centerline, G=construction, B=muscle. White means curve center; zero is >=6mm.
No camera-space decals, generated anatomy images, wireframe or noise.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image

P=argparse.ArgumentParser();P.add_argument('--assets',required=True);P.add_argument('--id',required=True);P.add_argument('--resolution',type=int,default=4096);a=P.parse_args();root=Path(a.assets);N=a.resolution
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

def path(name,points,channel=2,side='front',mirror=True):
 if channel==0 and side=='both':
  path(name,points,channel,'front',mirror);path(name,points,channel,'back',mirror);return
 for sign in ([1,-1] if mirror else [1]):
  pts=np.array(points,dtype=np.float32);pts[:,0]*=sign
  # Match rest bone stations; women have a shorter upper arm, longer forearm.
  if a.id=='female':
   xx=np.abs(pts[:,0]);pts[:,0]=np.sign(pts[:,0])*np.interp(xx,[0,.165,.453,.701,.9],[0,.167,.420,.693,.9])
  s=smooth(pts)
  front=(Y<.025) if side=='front' else (Y>.025) if side=='back' else np.ones_like(Y,dtype=bool)
  select=front&(X>=s[:,0].min()-radius)&(X<=s[:,0].max()+radius)&(Z>=s[:,1].min()-radius)&(Z<=s[:,1].max()+radius)
  if channel in (0,2):
   s=project_to_surface(s,side);finite=np.isfinite(s[:,0])
   if finite.sum()<2:continue
   good=s[finite]
   select=(X>=good[:,0].min()-radius)&(X<=good[:,0].max()+radius)&(Z>=good[:,2].min()-radius)&(Z<=good[:,2].max()+radius)&(Y>=good[:,1].min()-radius)&(Y<=good[:,1].max()+radius)
  idx=np.flatnonzero(select);v=samples[idx] if channel in (0,2) else np.column_stack([X[idx],Z[idx]]);dist=np.full(len(idx),radius,np.float32)
  for p,q in zip(s,s[1:]):
   if not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q-p)>.035:continue
   d=q-p;t=np.clip(((v-p)*d).sum(1)/max(float(d@d),1e-12),0,1);dist=np.minimum(dist,np.linalg.norm(v-(p+t[:,None]*d),axis=1))
  masks[idx,channel]=np.maximum(masks[idx,channel],np.clip(1-dist/radius,0,1))
  paths.append({'name':name+(' L' if sign==1 else ' R') if mirror else name,'channel':channel,'surface':side,'points':pts.tolist()})

# Sparse cross sections at anatomical stations, plus surface meridians.
path('sagittal center',[(0,.78),(0,.94),(0,1.15),(0,1.44),(0,1.58),(0,1.79)],0,'both',False)
for zz,name in [(1.657,'eye level'),(1.492,'neck base')]:path(name,[(-.13,zz),(0,zz),(.13,zz)],1,'both',False)
for zz,name in [(1.365,'chest upper'),(1.270,'chest lower'),(1.135,'waist'),(1.025,'pelvis')]:
 path(name,[(-.18,zz),(-.09,zz-.006),(0,zz-.009),(.09,zz-.006),(.18,zz)],1,'both',False)
path('torso volume meridian',[(.08,1.46),(.115,1.36),(.113,1.27),(.093,1.14),(.108,1.03),(.112,.95),(.108,.82),(.101,.67),(.093,.514),(.107,.39),(.093,.26),(.084,.105),(.073,.03)],1,'both')
path('arm axis',[(.10,1.466),(.16,1.457),(.205,1.431),(.32,1.410),(.453,1.401),(.58,1.396),(.70,1.394),(.775,1.383)],0,'both')
for xx,name in [(.30,'upper arm section'),(.453,'elbow'),(.58,'forearm section'),(.70,'wrist')]:path(name,[(xx,1.31),(xx,1.50)],1,'both')
path('leg axis',[(.078,.93),(.075,.78),(.072,.514),(.073,.31),(.073,.105)],0,'both')
for zz,name in [(.78,'thigh section'),(.514,'knee'),(.31,'calf section'),(.105,'ankle')]:path(name,[(.015,zz),(.17,zz)],1,'both')

# Front: neck tendons, pectoral arc, deltoid cap, rectus and oblique borders.
for name,pts in [
 ('sternocleidomastoid',[(.045,1.54),(.045,1.49),(.018,1.44)]),
 ('clavicle',[(.012,1.438),(.06,1.441),(.112,1.448),(.164,1.435)]),
 ('pectoral lower border',[(.012,1.315),(.052,1.298),(.104,1.299),(.15,1.328),(.18,1.378)]),
 ('pectoral outer border',[(.164,1.435),(.181,1.412),(.177,1.378)]),
 ('deltoid insertion',[(.171,1.457),(.22,1.450),(.277,1.417),(.281,1.385),(.235,1.357)]),
 ('biceps upper',[(.274,1.422),(.335,1.430),(.393,1.419),(.433,1.403)]),
 ('biceps lower',[(.277,1.378),(.334,1.370),(.395,1.380),(.433,1.399)]),
 ('brachioradialis',[(.474,1.418),(.524,1.412),(.575,1.400),(.665,1.394)]),
 ('forearm flexor',[(.474,1.378),(.530,1.366),(.591,1.376),(.671,1.390)]),
 ('rectus border',[(.045,1.288),(.051,1.239),(.045,1.18),(.036,1.12),(.022,1.074)]),
 ('external oblique',[(.142,1.283),(.128,1.232),(.117,1.179),(.105,1.116),(.119,1.065)]),
 ('iliac / inguinal',[(.143,1.057),(.114,1.028),(.073,1.007),(.027,.976)]),
 ('quadriceps lateral',[(.129,.961),(.144,.875),(.134,.746),(.115,.615),(.092,.551)]),
 ('quadriceps medial',[(.043,.925),(.031,.815),(.034,.698),(.043,.602),(.059,.551)]),
 ('sartorius direction',[(.127,.969),(.098,.866),(.068,.740),(.045,.603)]),
 ('patella',[(.055,.539),(.076,.548),(.096,.530),(.092,.506),(.074,.492),(.054,.512)]),
 ('tibial crest',[(.073,.481),(.076,.394),(.071,.282),(.072,.133)]),
 ('calf lateral',[(.103,.475),(.120,.399),(.111,.322),(.093,.239),(.077,.133)]),
 ('calf medial',[(.047,.475),(.027,.398),(.037,.320),(.058,.237)])]:path(name,pts)
path('linea alba',[(0,1.42),(0,1.30),(0,1.20),(0,1.07)],2,'front',False)
for zz in [1.246,1.188,1.131]:path('rectus tendinous intersection',[(.005,zz),(.023,zz-.004),(.045,zz+.002)])
# Back: trapezius, scapula, lats, gluteal and posterior limb direction.
for name,pts in [
 ('trapezius',[(.025,1.53),(.072,1.478),(.162,1.445),(.109,1.393),(.035,1.292)]),
 ('scapula',[(.064,1.421),(.128,1.406),(.157,1.356),(.113,1.295),(.067,1.326)]),
 ('latissimus',[(.177,1.337),(.161,1.258),(.126,1.184),(.066,1.105)]),
 ('erector spinae',[(.025,1.371),(.033,1.279),(.033,1.178),(.025,1.09)]),
 ('posterior deltoid',[(.173,1.455),(.23,1.445),(.286,1.413),(.253,1.362)]),
 ('triceps',[(.276,1.412),(.330,1.414),(.390,1.397),(.435,1.397)]),
 ('triceps lower',[(.274,1.369),(.332,1.366),(.399,1.381)]),
 ('forearm extensor',[(.477,1.421),(.546,1.407),(.671,1.394)]),
 ('gluteus outline',[(.018,1.076),(.075,1.10),(.137,1.068),(.153,1.023),(.112,.977),(.051,.972),(.019,1.008)]),
 ('hamstring lateral',[(.131,.956),(.136,.837),(.112,.690),(.095,.558)]),
 ('hamstring medial',[(.040,.949),(.028,.83),(.042,.686),(.057,.563)]),
 ('popliteal fold',[(.049,.522),(.072,.510),(.10,.526)]),
 ('gastrocnemius lateral',[(.083,.482),(.113,.436),(.116,.368),(.092,.309),(.077,.256)]),
 ('gastrocnemius medial',[(.069,.482),(.036,.439),(.026,.366),(.05,.308),(.071,.255)]),
 ('Achilles',[(.071,.258),(.071,.177),(.072,.108)])]:path(name,pts,2,'back')
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
for name,base,guide,muscle in [('gray-guide',gray,True,False),('flesh',[221,183,161],False,False),('flesh-guide',[221,183,161],True,False),('muscle',[211,195,177],False,True)]:
 color=np.empty((N,N,3),np.float32);color[:]=base
 def ink(ch,col,width,opacity):
  v=mask[:,:,ch].astype(np.float32)/255;coverage=np.clip((v-(1-width/radius))/.13+.5,0,1)*opacity
  color[:]=color*(1-coverage[:,:,None])+np.array(col)*coverage[:,:,None]
 if guide:ink(1,[91,100,110],.00135,.62);ink(0,[255,138,53],.0022,1.0)
 if muscle:ink(2,[66,51,46],.00165,.9)
 Image.fromarray(np.clip(color,0,255).astype(np.uint8)).save(root/(a.id+'-'+name+'.png'))
print('TEXTURES READY',a.id,flush=True)
