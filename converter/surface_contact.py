"""Experimental surface adherence. Contact is additional to full-mesh safety.

A frozen hand patch and an explicit body region prevent nearest-body snapping
from switching effectors or treating a penetrating vertex as successful touch.
Distances are point-to-triangle, in metres, not joint/vertex-to-vertex distances.
"""
import hashlib
import numpy as np

VERSION = 'surface-contact-experiment-v1'
TARGET_GAP_M = .002
MAX_GAP_M = .005
MIN_CLEARANCE_M = .0005
MIN_PATCH_AREA_M2 = 25e-6


def nearest_surface(points, triangles):
    points=np.asarray(points,float);triangles=np.asarray(triangles,float)
    if (points.ndim!=2 or points.shape[1]!=3 or triangles.ndim!=3 or triangles.shape[1:]!=(3,3)
            or not len(points) or not len(triangles) or not np.isfinite(points).all() or not np.isfinite(triangles).all()):
        raise ValueError('contact_invalid_geometry')
    a,b,c=triangles.transpose(1,0,2);ab=b-a;ac=c-a
    normal=np.cross(ab,ac);length=np.linalg.norm(normal,axis=1)
    if np.any(length<1e-12):raise ValueError('contact_degenerate_target')
    normal/=length[:,None];out=[];ids=[];dist=[];signed=[]
    for p in points:
        height=np.einsum('ij,ij->i',p-a,normal);projected=p-height[:,None]*normal
        d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac)
        ap=projected-a;d20=np.einsum('ij,ij->i',ap,ab);d21=np.einsum('ij,ij->i',ap,ac)
        denom=d00*d11-d01*d01;u=(d11*d20-d01*d21)/denom;v=(d00*d21-d01*d20)/denom
        candidates=[projected];valid=(u>=0)&(v>=0)&(u+v<=1)
        ds=[np.where(valid,height**2,np.inf)]
        for start,end in ((a,b),(b,c),(c,a)):
            edge=end-start;t=np.clip(np.einsum('ij,ij->i',p-start,edge)/np.einsum('ij,ij->i',edge,edge),0,1)
            q=start+t[:,None]*edge;candidates.append(q);ds.append(np.sum((p-q)**2,axis=1))
        which,idx=np.unravel_index(np.argmin(ds),np.shape(ds));q=candidates[which][idx]
        out.append(q);ids.append(idx);dist.append(np.linalg.norm(p-q));signed.append(np.dot(p-q,normal[idx]))
    return np.array(out),np.array(ids),np.array(dist),np.array(signed),normal[np.array(ids)]


def topology_hash(faces):
    return hashlib.sha256(np.asarray(faces,dtype='<i8').tobytes()).hexdigest()


def define_contact(vertices,faces,weights,names,side,bones):
    if side not in ('Left','Right'):raise ValueError('contact_invalid_side')
    names=[str(n).split(':')[-1] for n in names];faces=np.asarray(faces,int)
    palm=weights[:,[i for i,n in enumerate(names) if n.startswith(side+'Hand')]].sum(1)>.8
    # Pelvis skin blends into the upper-leg groups. Using Hips alone
    # truncates its physical surface at the groin and creates a false edge.
    hip=bones[names.index('Hips'),:3,3]
    shoulder=(bones[names.index('LeftArm'),:3,3]+bones[names.index('RightArm'),:3,3])*.5
    up=shoulder-hip;torso_length=np.linalg.norm(up);up/=torso_length
    height=(vertices-hip)@up
    torso=(weights[:,[names.index(n) for n in ('Hips','Spine','LeftUpLeg','RightUpLeg')]].sum(1)>.8)&(height>-.35*torso_length)&(height<.35*torso_length)
    target=np.flatnonzero(np.all(torso[faces],axis=1));hand=np.flatnonzero(np.all(palm[faces],axis=1))
    if not len(target) or not len(hand):raise ValueError('contact_skin_regions_missing')
    centers=vertices[faces[hand]].mean(1)
    _,_,distance,_,_=nearest_surface(centers,vertices[faces[target]])
    seed=int(hand[np.argmin(distance)])
    # Freeze a connected, local skin patch before optimization. A point alone
    # cannot establish adherence, and the optimizer may not reselect a winner.
    selected={seed}
    for _ in range(6):
        active=np.unique(faces[list(selected)])
        selected.update(int(i) for i in hand if np.any(np.isin(faces[i],active)))
        triangles=vertices[faces[list(selected)]]
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()*.5
        if area>=100e-6:break
    return dict(version=VERSION,side=side,target_region='pelvic_skin_including_proximal_thigh_blend',
        intent_authority='explicit_local_experiment_not_observed_ground_truth',
        effector_region=names[int(np.argmax(weights[faces[seed]].mean(0)))],
        topology_sha256=topology_hash(faces),vertex_count=len(vertices),
        hand_faces=sorted(selected),target_faces=target.tolist(),
        target_gap_m=TARGET_GAP_M,max_gap_m=MAX_GAP_M,min_clearance_m=MIN_CLEARANCE_M,
        min_patch_area_m2=MIN_PATCH_AREA_M2)


def contact_samples(vertices,faces,contract):
    if contract.get('version')!=VERSION or contract.get('topology_sha256')!=topology_hash(faces) or contract.get('vertex_count')!=len(vertices):
        raise ValueError('contact_identity_mismatch')
    hf=np.asarray(contract['hand_faces'],int);tf=np.asarray(contract['target_faces'],int)
    if len(np.unique(hf))!=len(hf) or len(np.unique(tf))!=len(tf):raise ValueError('contact_duplicate_faces')
    if not len(hf) or not len(tf) or np.any(hf<0) or np.any(tf<0) or max(hf.max(),tf.max())>=len(faces) or np.intersect1d(hf,tf).size:
        raise ValueError('contact_region_invalid')
    tri=vertices[faces[hf]];centers=tri.mean(1)
    q,ids,d,s,n=nearest_surface(centers,vertices[faces[tf]])
    areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
    return centers,q,d,s,n,areas


def audit_contact(vertices,faces,contract,*,mesh_validated=False):
    centers,q,d,s,n,areas=contact_samples(vertices,faces,contract)
    # Hard constants are not configurable by a supplied contract.
    near=(d<=MAX_GAP_M)&(s>=MIN_CLEARANCE_M)
    vertices_ids=np.unique(faces[contract['hand_faces']])
    _,_,vd,vs,_=nearest_surface(vertices[vertices_ids],vertices[faces[contract['target_faces']]])
    sample_index={int(vertex):i for i,vertex in enumerate(vertices_ids)}
    corners=np.array([[sample_index[int(vertex)] for vertex in face] for face in faces[contract['hand_faces']]])
    near &= np.all((vd[corners]<=MAX_GAP_M)&(vs[corners]>=MIN_CLEARANCE_M),axis=1)
    area=float(areas[near].sum());fraction=area/float(areas.sum())
    reasons=[]
    if not mesh_validated:reasons.append('full_mesh_safety_not_verified')
    if np.min(vs)<MIN_CLEARANCE_M:reasons.append('patch_clearance_below_minimum')
    if area<MIN_PATCH_AREA_M2 or fraction<.20:reasons.append('insufficient_contact_area')
    return dict(version=VERSION,status='contact_validated' if not reasons else 'contact_blocked',
        reasons=reasons,target_gap_m=TARGET_GAP_M,max_gap_m=MAX_GAP_M,min_clearance_m=MIN_CLEARANCE_M,
        minimum_sample_gap_m=float(min(d.min(),vd.min())),median_patch_gap_m=float(np.median(d)),maximum_patch_sample_gap_m=float(max(d.max(),vd.max())),
        area_measurement='sum_of_faces_with_all_corners_and_centroid_in_contact_band',
        minimum_signed_clearance_m=float(vs.min()),contact_area_mm2=area*1e6,
        patch_area_mm2=float(areas.sum())*1e6,contact_area_fraction=fraction,
        full_mesh_validated=bool(mesh_validated),mathematical_zero_contact=False,
        anchors=dict(hand=centers[np.argmin(d)].tolist(),body=q[np.argmin(d)].tolist()))
