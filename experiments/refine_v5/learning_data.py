"""P2/P3 supervised examples. Native GT is accessible only in data building.

Synthetic mesh projections use known cameras; real camera is a source-frame
hypothesis. Frozen features below are engineered HOG, not pretrained embeddings.
"""
import hashlib, json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

BONES=['LeftArm','LeftForeArm','RightArm','RightForeArm','LeftUpLeg','LeftLeg','RightUpLeg','RightLeg','Head','LeftHand','RightHand','LeftFoot','RightFoot']
PARTS=['left_upper_arm','left_forearm','right_upper_arm','right_forearm','left_thigh','left_shin','right_thigh','right_shin','head','left_hand','right_hand','left_foot','right_foot']
POINTS=['LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand','LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg','RightLeg','RightFoot']
COCO=[5,7,9,6,8,10,11,13,15,12,14,16]
REPR=['outlined','illustration','rendered','photo','stylized','stick']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def frame(bones):
    by={b['name'].split(':')[-1]:b for b in bones}
    def head(n):return np.array(by[n].get('head',np.array(by[n]['matrix'])[:3,3]),float)
    center=head('Hips');up=head('Spine2')-center;scale=np.linalg.norm(up)
    if scale<1e-6:raise ValueError('degenerate torso')
    up/=scale;right=head('LeftArm')-head('RightArm');right-=up*np.dot(right,up)
    if np.linalg.norm(right)<1e-6:raise ValueError('degenerate shoulders')
    right/=np.linalg.norm(right);forward=np.cross(right,up)
    return by,center,scale,np.column_stack([right,up,forward])

def source_camera(bones,view):
    # Reuse the library's view angle; native FBX is Z-up, BVH/search is Y-up.
    from src.library import view_angle, _rot_y
    _,_,_,body=frame(bones)
    native_to_search=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    image_camera=np.diag([1.,-1.,-1.])@_rot_y(view_angle(view))@native_to_search
    return image_camera@body

def axes(bones):
    by,c,s,f=frame(bones)
    a=np.array([np.array(by[n]['matrix'])[:3,:2].T@f for n in BONES])
    a/=np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-8)
    return a.astype('float32')

def visibility(review):
    if review['target_identity']=='ambiguous':return np.zeros(13,'float32')
    v=review['part_visibility'];m=np.array([v[p]=='visible' for p in PARTS],dtype='float32')
    # Endpoints must also belong to an observable upstream segment.
    for child,parent in [(1,0),(3,2),(5,4),(7,6),(9,1),(10,3),(11,5),(12,7)]:m[child]*=m[parent]
    return m

def point_visibility(part_mask):
    # Shared elbow/knee belongs to the visible parent even if the child is hidden.
    m=np.asarray(part_mask,bool);out=[]
    for start in [0,2,4,6]:out.extend([m[start],m[start],m[start] and m[start+1]])
    return np.asarray(out,bool)

def normalized_points(kp,scores):
    q=np.array(kp,float)[COCO];score=np.array(scores,float)[COCO]
    center=q[[0,3]].mean(0);width=np.linalg.norm(q[0]-q[3])
    if width<1e-6 or not np.isfinite(q).all():raise ValueError('degenerate observation')
    return ((q-center)/width).astype('float32'),(score>=.3).astype('float32')

def image_features(gray,points):
    gray=np.asarray(gray,np.float32)/255.;small=cv2.resize(gray,(32,32))
    dx=cv2.Sobel(small,cv2.CV_32F,1,0);dy=cv2.Sobel(small,cv2.CV_32F,0,1)
    mag,ang=cv2.cartToPolar(dx,dy);bins=(ang*8/(2*np.pi)).astype(int)%8
    cells=[]
    for y in range(0,32,8):
        for x in range(0,32,8):
            h=np.bincount(bins[y:y+8,x:x+8].ravel(),weights=mag[y:y+8,x:x+8].ravel(),minlength=8)
            cells.extend(h/(np.linalg.norm(h)+1e-5))
    h,w=gray.shape;patches=[]
    for p in points:
        x,y=np.round(p).astype(int);r=max(2,int(min(h,w)*.09))
        pch=gray[max(0,y-r):min(h,y+r+1),max(0,x-r):min(w,x+r+1)]
        patches.extend(cv2.resize(pch,(4,4)).ravel() if pch.size else np.ones(16))
    return np.r_[cells,patches].astype('float32')

def geometry(base,q,qmask,rep,camera):
    return np.r_[base.ravel(),(q*qmask[:,None]).ravel(),qmask,[float(rep==r) for r in REPR],np.asarray(camera).ravel()].astype('float32')

def _render(vertices,faces,heads,rotation,rng,randomized):
    v=vertices@rotation.T;h=heads@rotation.T
    # camera coords x,z -> image x,-y; all geometry remains one known camera.
    p=v[:,[0,2]]*np.array([1,-1]);q=h[:,[0,2]]*np.array([1,-1])
    lo=p.min(0);hi=p.max(0);scale=80/max((hi-lo).max(),1e-4)
    center=(lo+hi)/2;xy=(p-center)*scale+48;kp=(q-center)*scale+48
    silhouette=np.zeros((96,96),np.uint8)
    for tri in xy[faces].astype(np.int32):cv2.fillConvexPoly(silhouette,tri,255)
    edge=cv2.morphologyEx(silhouette,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    ink=edge.copy();observed=np.ones(12,'float32')
    if randomized:
        if rng.random()<.25:ink=silhouette if rng.random()<.5 else cv2.dilate(edge,np.ones((2,2),np.uint8))
        # Background stroke noise, blur, crop and rectangle occluder.
        for _ in range(rng.integers(0,8)):
            p1,p2=rng.integers(0,96,(2,2));cv2.line(ink,tuple(p1),tuple(p2),int(rng.integers(30,100)),1)
        if rng.random()<.35:
            x,y=rng.integers(0,70,2);ww,hh=rng.integers(12,40,2);ink[y:y+hh,x:x+ww]=0
            observed[((kp[:,0]>=x)&(kp[:,0]<x+ww)&(kp[:,1]>=y)&(kp[:,1]<y+hh))]=0
        if rng.random()<.25:
            bottom=int(rng.integers(45,85));ink[bottom:]=0;observed[kp[:,1]>=bottom]=0
        if rng.random()<.3:ink=cv2.GaussianBlur(ink,(3,3),.7)
    observed*=((kp>=0)&(kp<96)).all(1)
    return 255-ink,kp,observed

def build(run,previous,repo,variants=32):
    run,previous,repo=map(Path,(run,previous,repo));out=run/'dataset';out.mkdir(exist_ok=False)
    rigs=json.loads((run/'native-data/records.json').read_text())
    gt={r['case_id']:r for r in json.loads((previous/'data/native_audit.json').read_text())}
    seeds={r['case_id']:r for r in json.loads((previous/'all-seeds/seeds.json').read_text())}
    reviews={r['case_id']:r for r in json.loads((repo/'evaluation/refine_v5/visual_review_v2.json').read_text())['records']}
    split_rows=json.loads((repo/'evaluation/refine_v5/split-v2.json').read_text())['records']
    groups={r['case_id']:r['source_risk_group'] for r in split_rows}
    train_groups={r['source_risk_group'] for r in split_rows if r['split']=='development'}
    val_groups={r['source_risk_group'] for r in split_rows if r['split']=='validation'}
    if train_groups & val_groups:raise ValueError('source-group leakage')
    manifest=[];arrays=[];rng=np.random.default_rng(711)
    for rig in rigs:
        if rig['status']!='ready':continue
        case=rig['case_id'];review=reviews[case];seed=seeds[case]
        base=axes(rig['bones']);label=axes(gt[case]['bones']);mask=visibility(review)
        q,qmask=normalized_points(seed['keypoints'],seed['scores'])
        by,c,s,f=frame(rig['bones']);camera=source_camera(rig['bones'],seed['view'])
        gray=cv2.imread(seed['image'],cv2.IMREAD_GRAYSCALE)
        if gray is None:raise ValueError('missing image')
        coords=np.array(seed['keypoints'])[COCO]
        # Point-centred patches. Empty/cropped patches carry explicit masks.
        img=image_features(gray,coords)
        arrays.append((geometry(base,q,qmask,review['representation'],camera),img,base,label,mask,0.,case,rig['split'],'real'))
        mesh=json.loads(Path(rig['gt_mesh']).read_text());by,gc,gs,gf=frame(gt[case]['bones'])
        vertices=(np.array(mesh['vertices'])-gc)@gf/gs;faces=np.array(mesh['faces'],int)
        heads=np.array([(np.array(by[n]['head'])-gc)@gf/gs for n in POINTS])
        count=variants if rig['split']=='development' else 8
        for style in ('clean','randomized'):
            for k in range(count):
                # Canonical body has vertical y; renderer maps to Blender z up.
                yaw,pitch=rng.uniform(-np.pi,np.pi),rng.uniform(-.35,.35)
                cam=Rotation.from_euler('xy',[pitch,yaw]).as_matrix()
                rot=np.array([[1,0,0],[0,0,-1],[0,1,0]])@cam
                image,pix,obs=_render(vertices,faces,heads,rot,rng,style=='randomized')
                width=np.linalg.norm(pix[0]-pix[3])
                if width<2:continue
                norm=(pix-pix[[0,3]].mean(0))/width
                noisy=norm+rng.normal(0,.035,norm.shape)*(style=='randomized')
                # Source proposals near the target: bounded, known rotations.
                residual=Rotation.from_rotvec(rng.normal(0,.23,(13,3))).as_matrix()
                b=np.einsum('nij,nkj->nki',residual,label).astype('float32')
                imf=image_features(image,pix)
                arrays.append((geometry(b,noisy,obs,'outlined',rot),imf,b,label,np.ones(13,'float32'),1.,case,rig['split'],style))
                if k==0 and style=='randomized':cv2.imwrite(str(out/(case+'.synthetic.png')),image)
        manifest.append(dict(case_id=case,split=rig['split'],image_sha256=seed['image_sha256'],source_sha256=digest(rig['base_fbx']),gt_sha256=gt[case]['sha256'],source_risk_group=groups[case],real_camera='fixed_selected_library_view_GT_camera_unavailable',seed_kind=seed['seed_kind']))
    keys=['geometry','image','base','target','mask','synthetic','case_id','split','domain']
    data={k:np.array([row[i] for row in arrays]) for i,k in enumerate(keys)}
    np.savez_compressed(out/'examples.npz',**data)
    (out/'manifest.json').write_text(json.dumps(dict(schema='v5-p2-p3-supervised-v1',test_opened=False,labels='native_FBX',supervision='direction_strong_roll_weak_on_real',mesh_projection=True,synthetic_cameras='known',randomization=['camera','line_width','silhouette','background_strokes','blur','crop','occlusion','observation_noise'],frozen_image_features='engineered_HOG_and_part_patches_not_pretrained',records=manifest,examples=len(arrays)),indent=2))
    print('V5_DATASET',len(manifest),len(arrays))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--previous',required=True);p.add_argument('--repo',required=True);p.add_argument('--variants',type=int,default=32);a=p.parse_args();build(a.run,a.previous,a.repo,a.variants)
