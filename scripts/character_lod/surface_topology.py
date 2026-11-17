"""Bounded vertex-star deletion on all authored fields, seam/nail protection."""
import json
import bmesh
import numpy as np
from scripts.character_lod.contract import require_subset


def mask(mesh, arm, co, uv, faces, freeze_hand_graph=False):
    names = {g.index: g.name for g in mesh.vertex_groups}
    protected = np.zeros(len(co), bool)
    foot = np.zeros(len(co), bool)
    for v in mesh.data.vertices:
        protected[v.index] = sum(g.weight for g in v.groups
            if any(t in names[g.group] for t in ('Head','Neck'))) > .2
        # Entire distal phalanx (including nail support), not just a point sample.
        protected[v.index] |= any('Hand' in names[g.group] and
            names[g.group].endswith('3') and g.weight > .05 for g in v.groups)
        if freeze_hand_graph:
            protected[v.index] |= sum(g.weight for g in v.groups
                if 'Hand' in names[g.group]) > .01
        foot[v.index] = sum(g.weight for g in v.groups
            if any(t in names[g.group] for t in ('Foot','Toe'))) > .2
    for side in ('Left','Right'):
        for short, radius in [('Arm',.030),('ForeArm',.022),('Hand',.020),
                              ('UpLeg',.038),('Leg',.028),('Foot',.022)]:
            p = np.array(arm.data.bones['mixamorig:'+side+short].head_local)
            protected |= np.linalg.norm(co-p,axis=1) < radius
        for bone in arm.data.bones:
            if side+'Hand' in bone.name and any(x in bone.name for x in ('Thumb','Index','Middle','Ring','Pinky')):
                protected |= np.linalg.norm(co-np.array(bone.head_local),axis=1)<.006
        toe = arm.data.bones['mixamorig:'+side+'ToeBase']
        head, tail = np.array(toe.head_local), np.array(toe.tail_local)
        axis = tail-head; u = (co-head)@axis/(axis@axis)
        # Keep the distal toe/nail area intact; simplify flat midfoot/heel only.
        protected |= foot & (u>.45) & (np.linalg.norm(co-head,axis=1)<.16)
    lo=np.full((len(co),2),np.inf);hi=np.full((len(co),2),-np.inf)
    np.minimum.at(lo,faces.ravel(),uv.reshape(-1,2));np.maximum.at(hi,faces.ravel(),uv.reshape(-1,2))
    protected |= np.any(hi-lo>1e-6,axis=1)
    lm=json.loads(mesh['landmarks']);protected[[lm['chin_vertex'],lm['crown_vertex']]]=True
    # Existing height/floor gates require the exact support extrema, not a
    # tolerance-sized approximation of the sole plane.
    protected |= co[:,2] <= co[:,2].min()+.0005
    protected[np.r_[np.argmin(co,axis=0),np.argmax(co,axis=0)]]=True
    edges=np.array([e.vertices[:] for e in mesh.data.edges]);p=protected.copy()
    protected[edges[np.any(p[edges],axis=1)].ravel()]=True
    return protected


def simplify(mesh, protected, rounds, uv, tolerance, span, extra_fields=(), keep=None, cost_order=False):
    bm=bmesh.new();bm.from_mesh(mesh.data)
    layer=bm.verts.layers.int.new('source_index');uvlayer=bm.loops.layers.uv.new('source_uv')
    for v in bm.verts:v[layer]=v.index+1
    for f,values in zip(bm.faces,uv.reshape(-1,3,2)):
        for loop,value in zip(f.loops,values):loop[uvlayer].uv=value
    def coordinates(data):
        a=np.empty(len(data)*3);data.foreach_get('co',a);return a.reshape(-1,3)
    fields=np.stack([coordinates(k.data) for k in mesh.data.shape_keys.key_blocks
        if k.name=='Basis' or k.name.startswith(('BODY_','PSD_'))]+list(extra_fields))
    history=[]
    for iteration in range(rounds):
        bm.verts.ensure_lookup_table();chosen=[];blocked=set();candidates=[]
        for v in bm.verts:
            sid=v[layer]-1
            if protected[sid] or (keep is not None and keep[sid]) or v in blocked or not v.is_manifold:continue
            ring=[e.other_vert(v) for e in v.link_edges]
            if len(ring)<3 or len(ring)>16 or any(protected[u[layer]-1] for u in ring):continue
            ri=np.array([u[layer]-1 for u in ring]);points=fields[:,ri,:]
            center=points.mean(1);_,_,vt=np.linalg.svd(points-center[:,None,:],full_matrices=False)
            departure=np.abs(np.einsum('ki,ki->k',fields[:,sid]-center,vt[:,-1,:]))
            limit = tolerance[sid] if np.ndim(tolerance) else tolerance
            if departure.max()>limit:continue
            if np.linalg.norm(points-fields[:,sid,None,:],axis=2).max()>span:continue
            if cost_order:
                candidates.append((float(departure.max()/limit),sid,v,ring))
            else:
                chosen.append(v);blocked.add(v);blocked.update(ring)
        if cost_order:
            for _,_,v,ring in sorted(candidates,key=lambda row:(row[0],row[1])):
                if v in blocked:continue
                chosen.append(v);blocked.add(v);blocked.update(ring)
        if not chosen:break
        bmesh.ops.dissolve_verts(bm,verts=chosen,use_face_split=False,use_boundary_tear=False)
        bmesh.ops.triangulate(bm,faces=[f for f in bm.faces if len(f.verts)>3],quad_method='BEAUTY',ngon_method='BEAUTY')
        history.append(len(chosen));print('SURFACE_ROUND',iteration+1,'removed',len(chosen),'remaining',len(bm.verts),flush=True)
    bm.verts.index_update();ids=np.array([v[layer]-1 for v in bm.verts])
    faces=np.array([[v.index for v in f.verts] for f in bm.faces],np.int32)
    result_uv=np.array([[l[uvlayer].uv[:] for l in f.loops] for f in bm.faces]);bm.free()
    require_subset(ids,np.flatnonzero(protected),[p.vertices[:] for p in mesh.data.polygons],faces)
    if keep is not None and not set(np.flatnonzero(keep)).issubset(ids):
        raise ValueError('required surface-support vertex removed')
    return ids,faces,result_uv,history
