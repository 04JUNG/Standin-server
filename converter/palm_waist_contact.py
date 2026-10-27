"""Full-palm/side-waist contract. A thumb patch cannot satisfy this contract."""
import numpy as np
from converter.surface_contact import nearest_surface,topology_hash,MIN_CLEARANCE_M,MAX_GAP_M
from converter.hand_geometry import hand_frame
VERSION='full-palm-side-waist-v1'


def anatomical_axes(bones,names):
    lookup={str(n).split(':')[-1]:i for i,n in enumerate(names)}
    pos=lambda n:np.asarray(bones[lookup[n],:3,3],float)
    hip=pos('Hips');l=pos('LeftArm');r=pos('RightArm')
    lateral=l-r;lateral/=np.linalg.norm(lateral)
    up=(l+r)*.5-hip;up-=up.dot(lateral)*lateral;length=np.linalg.norm(up);up/=length
    front=np.cross(lateral,up)
    return hip,lateral,up,front,length


def palm_frame(bones,names,side):
    lookup={str(n).split(':')[-1]:i for i,n in enumerate(names)}
    wrist=bones[lookup[side+'Hand'],:3,3]
    bases={f:bones[lookup[side+'Hand'+f.title()+'1'],:3,3] for f in ('thumb','index','middle','ring','pinky')}
    return wrist,hand_frame(side.lower(),wrist,bases),np.linalg.norm(bases['middle']-wrist)


def define_palm_contract(vertices,faces,weights,names,bones,side):
    if side not in ('Left','Right'):raise ValueError('palm_side_invalid')
    names=[str(n).split(':')[-1] for n in names];f=np.asarray(faces,int)
    tri=vertices[f];center=tri.mean(1);normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(normal,axis=1)*.5
    normal/=np.maximum(2*area[:,None],1e-20)
    wrist,frame,length=palm_frame(bones,names,side)
    long=(center-wrist)@frame.length_axis/length;across=(center-wrist)@frame.pinky_side
    hand_weight=weights[:,names.index(side+'Hand')]
    palm=(hand_weight[f].min(1)>.55)&(normal@frame.palm_normal>.45)&(long>.12)&(long<1.03)
    selected=np.flatnonzero(palm)
    if len(selected)<12 or area[selected].sum()<500e-6:raise ValueError('full_palm_surface_missing')
    # Four fixed quadrants over a broad anatomical palm, excluding all digits.
    lateral_mid=np.median(across[selected]);long_mid=np.median(long[selected])
    zones=(long[selected]>long_mid).astype(int)*2+(across[selected]>lateral_mid).astype(int)
    if set(zones)!=set(range(4)):raise ValueError('full_palm_zones_missing')
    hip,lateral,up,front,torso=anatomical_axes(bones,names);outward=lateral if side=='Left' else -lateral
    height=(center-hip)@up;sidepos=(center-hip)@outward
    body=(weights[:,[names.index(n) for n in ('Hips','Spine','Spine1',side+'UpLeg')]].sum(1)>.8)
    waist=body[f].all(1)&(height>-.2*torso)&(height<.5*torso)&(sidepos>.15*torso)&(normal@outward>.4)
    target=np.flatnonzero(waist)
    if not len(target):raise ValueError('side_waist_surface_missing')
    return dict(version=VERSION,side=side,topology_sha256=topology_hash(f),vertex_count=len(vertices),palm_faces=selected.tolist(),zones=zones.tolist(),waist_faces=target.tolist(),
        intent_authority='user_requested_palm_side_waist_and_rearward_elbow',observed_depth_ground_truth=False,
        target_gap_m=.002,max_gap_m=MAX_GAP_M,min_clearance_m=MIN_CLEARANCE_M,required_area_fraction=.8,required_zone_fraction=.6,required_opposing_normal_cosine=.7)


def measure_palm(vertices,faces,contract,mesh_validated=False):
    if contract['version']!=VERSION or topology_hash(faces)!=contract['topology_sha256'] or len(vertices)!=contract['vertex_count']:raise ValueError('palm_contract_identity')
    pf=np.asarray(contract['palm_faces'],int);tf=np.asarray(contract['waist_faces'],int);zones=np.asarray(contract['zones'])
    if not len(pf) or not len(tf) or np.any(pf<0) or np.any(tf<0) or max(pf.max(),tf.max())>=len(faces) or np.intersect1d(pf,tf).size:raise ValueError('palm_contract_regions')
    if len(set(pf))!=len(pf) or len(set(tf))!=len(tf) or len(pf)!=len(zones) or set(zones)!=set(range(4)):raise ValueError('palm_contract_regions')
    tri=vertices[faces[pf]];centers=tri.mean(1);norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);areas=np.linalg.norm(norm,axis=1)*.5
    if np.any(areas<=1e-12):raise ValueError('palm_degenerate_surface')
    norm/=2*areas[:,None]
    q,_,d,s,n=nearest_surface(centers,vertices[faces[tf]])
    cosine=-np.einsum('ij,ij->i',norm,n)
    ids=np.unique(faces[pf]);_,_,vd,vs,_=nearest_surface(vertices[ids],vertices[faces[tf]])
    mapping={int(v):i for i,v in enumerate(ids)};corners=np.array([[mapping[int(v)] for v in face] for face in faces[pf]])
    near=(d<=MAX_GAP_M)&(s>=MIN_CLEARANCE_M)&(cosine>=.7)&np.all((vd[corners]<=MAX_GAP_M)&(vs[corners]>=MIN_CLEARANCE_M),axis=1)
    total=areas.sum();fraction=float(areas[near].sum()/total)
    regions=[dict(zone=z,area_mm2=float(areas[zones==z].sum()*1e6),contact_fraction=float(areas[near&(zones==z)].sum()/areas[zones==z].sum()),median_gap_mm=float(np.median(d[zones==z])*1000)) for z in range(4)]
    failures=[]
    if not mesh_validated:failures.append('full_mesh_not_verified')
    if total<500e-6:failures.append('palm_surface_too_small')
    if fraction<.8 or any(r['contact_fraction']<.6 for r in regions):failures.append('broad_palm_contact_missing')
    if vs.min()<MIN_CLEARANCE_M:failures.append('palm_clearance_violation')
    return dict(version=VERSION,status='palm_contact_validated' if not failures else 'palm_contact_blocked',reasons=failures,
        palm_area_mm2=float(total*1e6),contact_area_fraction=fraction,minimum_sample_gap_mm=float(min(d.min(),vd.min())*1000),median_gap_mm=float(np.median(d)*1000),maximum_sample_gap_mm=float(max(d.max(),vd.max())*1000),
        mean_opposing_normal_cosine=float(np.average(cosine,weights=areas)),zones=regions,full_mesh_validated=mesh_validated)
