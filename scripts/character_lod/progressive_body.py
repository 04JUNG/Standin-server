"""Offline body simplification across measured source deformations.

Non-body points and the complete hand smoothing graph remain frozen. Movable
body points use fixed barycentric maps on original body triangles; controls,
PSDs and skin weights share that map. No post-export simplification.
"""
import heapq
import json
import time
from pathlib import Path
import numpy as np


def hybrid_seed(original_faces, original_uv, seed_ids, seed_faces, seed_uv, labels):
    body=np.isin(labels,[3,4,5]);seed_global=seed_ids[seed_faces];fixed=~body[seed_global].all(1);scale=len(labels)
    def edge_data(f):
        edges=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1);codes=edges[:,0].astype(np.int64)*scale+edges[:,1];unique,inverse,counts=np.unique(codes,return_inverse=True,return_counts=True)
        if np.any(counts!=2):raise ValueError('source is not a closed manifold')
        face_ids=np.tile(np.arange(len(f)),3);adj=face_ids[np.argsort(inverse,kind='stable')].reshape(-1,2)
        return unique,adj
    source_edges,source_adj=edge_data(original_faces);seed_edges,seed_adj=edge_data(seed_global);common=np.isin(seed_edges,source_edges)
    # Grow the frozen seed patch until its cut consists only of original edges.
    while True:
        boundary=fixed[seed_adj[:,0]]^fixed[seed_adj[:,1]];bad=boundary&~common
        if not bad.any():break
        fixed[seed_adj[bad].ravel()]=True
    barrier=np.isin(source_edges,seed_edges[boundary]);inside=~body[original_faces].all(1);neighbors=source_adj[~barrier]
    while True:
        before=int(inside.sum());hit=inside[neighbors[:,0]]|inside[neighbors[:,1]];inside[neighbors[hit].ravel()]=True
        if int(inside.sum())==before:break
    global_faces=np.concatenate((original_faces[~inside],seed_global[fixed]));uv=np.concatenate((original_uv.reshape(-1,3,2)[~inside],seed_uv.reshape(-1,3,2)[fixed]))

    ids=np.unique(global_faces);inverse=np.full(len(labels),-1,np.int32);inverse[ids]=np.arange(len(ids));faces=inverse[global_faces]
    if not np.array_equal(np.sort(ids[~body[ids]]),np.sort(seed_ids[~body[seed_ids]])):raise ValueError('hybrid changed fixed non-body point set')
    edges=np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1);_,counts=np.unique(edges,axis=0,return_counts=True)
    if np.any(counts!=2):raise ValueError('hybrid boundary is not a closed manifold')
    from scripts.character_lod.contract import require_subset
    require_subset(ids,seed_ids[~body[seed_ids]],seed_ids[seed_faces],faces)
    return ids,faces,uv


def collapse(ids,faces,uv,fields,pinned,graph,target,checkpoints=(),max_edge=.05,on_checkpoint=None,geometry_pairs=(),importance=None,placement="endpoint",normalize_cost=0.,cost_frames=None,projector=None,surface_cost_ratio=1.02,frame_weights=None):
    if placement not in ("endpoint","edge","surface"):raise ValueError("unknown placement policy")
    """fields: frame x vertex x XYZ. Pinned coordinates cannot be removed."""
    ids=np.asarray(ids,np.int32);faces=faces.copy();uv=uv.copy();fields=np.asarray(fields,float);n=len(ids);nf=len(faces);d=len(fields)
    if fields.shape[1:]!=(n,3):raise ValueError('field shape does not match seed')
    alive=np.ones(n,bool);f_alive=np.ones(nf,bool);revision=np.zeros(n,np.int32);vfaces=[set() for _ in range(n)];links=[set() for _ in range(n)]
    for i,(a,b,c) in enumerate(faces):
        vfaces[a].add(i);vfaces[b].add(i);vfaces[c].add(i);links[a].update((b,c));links[b].update((a,c));links[c].update((a,b))
    active_faces=np.flatnonzero((~graph[faces]).any(1));mass=np.zeros(n)
    for k in range(3):np.add.at(mass,faces[active_faces,k],1.)
    quad=np.zeros((n,d,10),float)
    for frame,co in enumerate(fields):
        if cost_frames is not None and frame not in cost_frames:continue
        f=faces[active_faces];v=co[f];normal=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);length=np.linalg.norm(normal,axis=1)
        if (length<2e-12).any():raise ValueError('source field has a degenerate face')
        normal/=length[:,None];plane=np.c_[normal,-np.einsum('ni,ni->n',normal,v[:,0])];q=np.stack([plane[:,i]*plane[:,j] for i,j in ((0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3))],axis=1)
        if frame_weights is not None:q*=frame_weights[frame]
        if importance is not None:q*=np.asarray(importance)[f].mean(1)[:,None]
        for k in range(3):np.add.at(quad[:,frame],f[:,k],q)
    def coefficient(points):
        x,y,z=np.moveaxis(points,-1,0);return np.stack((x*x,2*x*y,2*x*z,2*x,y*y,2*y*z,2*y,z*z,2*z,np.ones_like(x)),axis=-1)
    # Endpoint coefficients are updated when a movable body point is projected.
    coeff=coefficient(fields.transpose(1,0,2));maps=[{int(i):1.} for i in ids] if placement in ("edge","surface") else None;owners={int(i):j for j,i in enumerate(ids)};heap=[];queued={};blocked={};serial=0
    def push_edges(edges):
        nonlocal serial
        es=np.array([e for e in edges if alive[e[0]] and alive[e[1]] and not (pinned[e[0]] and pinned[e[1]])],np.int32)
        if not len(es):return
        for batch in np.array_split(es,max(1,(len(es)+999)//1000)):
            a,b=batch.T;q=quad[a]+quad[b];ea=np.maximum(np.einsum('ndk,ndk->nd',q,coeff[a]),0).max(1);eb=np.maximum(np.einsum('ndk,ndk->nd',q,coeff[b]),0).max(1)
            normalization=np.maximum(mass[a]+mass[b],1.)**normalize_cost
            ea/=normalization;eb/=normalization
            ea[pinned[b]]=np.inf;eb[pinned[a]]=np.inf
            for u,v,ca,cb,norm in zip(a,b,ea,eb,normalization):
                u,v=int(u),int(v);key=(u,v);sig=(int(revision[u]),int(revision[v]));old=queued.get(key)
                if old==sig:continue
                queued[key]=sig
                candidates=[(u,v,float(ca),0.),(v,u,float(cb),1.)]
                if placement=='edge' and not pinned[u] and not pinned[v]:
                    # Minimize the maximum per-frame quadratic along this edge.
                    x=fields[:,u];delta=fields[:,v]-x;qq=quad[u]+quad[v]
                    x1=x+delta;xmid=x+delta*.5
                    c0=np.einsum('dk,dk->d',qq,coeff[u]);c1=np.einsum('dk,dk->d',qq,coeff[v]);cm=np.einsum('dk,dk->d',qq,coefficient(xmid))
                    aa=2*(c1+c0-2*cm);bb=c1-c0-aa;left=0.;right=1.
                    for iteration in range(12):
                        t1=(2*left+right)/3;t2=(left+2*right)/3
                        if np.max(aa*t1*t1+bb*t1+c0)<=np.max(aa*t2*t2+bb*t2+c0):right=t2
                        else:left=t1
                    t=(left+right)/2;value=float(max(0.,np.max(aa*t*t+bb*t+c0))/norm)
                    if value+1e-16<min(ca,cb):candidates.append((u,v,value,t))
                for keep,remove,cost,t in candidates:
                    if np.isfinite(cost):
                        serial+=1;heapq.heappush(heap,(cost,serial,u,v,*sig,keep,remove,t,True))
    edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    push_edges([tuple(map(int,e)) for e in edges if not graph[e].any()]);count=n;accepted=0;rejected=0;start=time.monotonic();outputs={};goals=sorted(set([target,*checkpoints]),reverse=True)
    while count>target and heap:
        cost,_,u,v,ru,rv,s,r,t,allow_project=heapq.heappop(heap);key=(u,v)
        if not alive[u] or not alive[v] or v not in links[u]:continue
        if (ru,rv)!=(int(revision[u]),int(revision[v])):continue
        queued.pop(key,None)
        if pinned[r]:continue
        edge_faces=vfaces[u]&vfaces[v]
        if len(edge_faces)!=2:rejected+=1;blocked[key]=True;continue
        opposing=set(faces[list(edge_faces)].ravel())-{u,v}
        if links[u]&links[v]!=opposing:rejected+=1;blocked[key]=True;continue
        projection=None
        if placement=='surface' and allow_project and not pinned[u] and not pinned[v]:
            qq=quad[u]+quad[v];x=fields[:,u];delta=fields[:,v]-x;c0=np.einsum('dk,dk->d',qq,coeff[u]);c1=np.einsum('dk,dk->d',qq,coeff[v]);cm=np.einsum('dk,dk->d',qq,coefficient(x+delta*.5));aa=2*(c1+c0-2*cm);bb=c1-c0-aa;left=0.;right=1.
            for iteration in range(12):
                t1=(2*left+right)/3;t2=(left+2*right)/3
                if np.max(aa*t1*t1+bb*t1+c0)<=np.max(aa*t2*t2+bb*t2+c0):right=t2
                else:left=t1
            tp=(left+right)/2;fi=next(iter(edge_faces));uv_u=uv[fi,np.flatnonzero(faces[fi]==u)[0]];uv_v=uv[fi,np.flatnonzero(faces[fi]==v)[0]]
            proposed=projector((1-tp)*fields[-1,u]+tp*fields[-1,v],maps[u],maps[v],tp,(1-tp)*uv_u+tp*uv_v,float(np.linalg.norm(uv_u-uv_v)))
            if proposed is not None:
                vertex_map,new_fields,new_uv=proposed;new_cost=float(max(0.,np.einsum('dk,dk->d',qq,coefficient(new_fields)).max())/(max(mass[u]+mass[v],1.)**normalize_cost))
                duplicate=len(vertex_map)==1 and owners.get(next(iter(vertex_map))) not in (None,u,v)
                if not duplicate and new_cost<=cost*surface_cost_ratio+1e-16:projection=proposed
        moving=(placement=='edge' and 0<t<1) or projection is not None
        newpoint=(1-t)*fields[:,u]+t*fields[:,v] if moving else fields[:,s]
        merged=None
        if projection is not None:merged,newpoint,projected_uv=projection
        elif moving:
            merged={key:value*(1-t) for key,value in maps[u].items()}
            for source_id,value in maps[v].items():merged[source_id]=merged.get(source_id,0.)+t*value
            merged={key:value for key,value in merged.items() if value>=1e-10};weight_mass=sum(merged.values());merged={key:value/weight_mass for key,value in merged.items()}
            if len(merged)>64:rejected+=1;blocked[key]=True;continue
        changed=np.array(sorted((vfaces[s]|vfaces[r] if moving else vfaces[r])-edge_faces),np.int32);before=faces[changed];after=before.copy();after[after==r]=s
        fan=(vfaces[s]|vfaces[r])-edge_faces;future=[tuple(sorted(after[list(changed).index(i)] if i in vfaces[r] else faces[i])) for i in fan]
        if len(set(future))!=len(future):rejected+=1;blocked[key]=True;continue
        old=fields[:,before];new=fields[:,after].copy()
        if moving:
            for fi,slot in np.argwhere(after==s):new[:,fi,slot]=newpoint
        on=np.cross(old[:,:,1]-old[:,:,0],old[:,:,2]-old[:,:,0]);nn=np.cross(new[:,:,1]-new[:,:,0],new[:,:,2]-new[:,:,0])
        valid=bool((np.einsum('dfi,dfi->df',on,nn)>0).all() and (np.linalg.norm(nn,axis=2)>=2e-12).all())
        if valid and geometry_pairs:
            a,b=np.asarray(geometry_pairs,np.int32).T
            valid=bool((np.einsum('dfi,dfi->df',nn[a],nn[b])>0).all())
        if valid and max_edge:
            valid=bool(max(np.linalg.norm(new[:,:,i]-new[:,:,j],axis=2).max() for i,j in ((0,1),(1,2),(2,0)))<=max_edge)
        euv=[];euv_r=[]
        for fi in edge_faces:
            euv.append(uv[fi,np.flatnonzero(faces[fi]==s)[0]])
            euv_r.append(uv[fi,np.flatnonzero(faces[fi]==r)[0]])
        if not valid or np.max(np.abs(euv[0]-euv[1]))>1e-6 or (moving and np.max(np.abs(euv_r[0]-euv_r[1]))>1e-6):
            if projection is not None:
                serial+=1;heapq.heappush(heap,(cost,serial,u,v,ru,rv,s,r,t,False))
            rejected+=1;blocked[key]=True;continue
        updated_uv=projected_uv if projection is not None else (1-t)*euv[0]+t*euv_r[0] if s==u else euv[0]
        affected=(links[s]|links[r])-{s,r}
        for fi in edge_faces:
            f_alive[fi]=False
            for vertex in faces[fi]:vfaces[vertex].discard(fi)
        for fi,oldf,newf in zip(changed,before,after):
            if moving:
                for slot in np.flatnonzero((oldf==r)|(oldf==s)):uv[fi,slot]=updated_uv
            else:
                for slot in np.flatnonzero(oldf==r):uv[fi,slot]=euv[0]
            faces[fi]=newf;vfaces[r].discard(fi);vfaces[s].add(fi)
        if maps is not None:
            for carrier in (s,r):
                if len(maps[carrier])==1:
                    original_id=next(iter(maps[carrier]))
                    if owners.get(original_id)==carrier:owners.pop(original_id)
        if moving:
            fields[:,s]=newpoint;coeff[s]=coefficient(newpoint);maps[s]=merged
        if maps is not None and len(maps[s])==1:owners[next(iter(maps[s]))]=s
        alive[r]=False;count-=1;accepted+=1;quad[s]+=quad[r];mass[s]+=mass[r]
        for vertex in affected:
            links[vertex].discard(r);links[vertex].add(s)
        links[s]=(links[s]|links[r])-{s,r};links[r].clear();vfaces[r].clear();revision[s]+=1;revision[r]+=1
        new_edges={tuple(sorted((s,int(x)))) for x in links[s] if alive[x] and not graph[x]}
        # A previously rejected edge can become valid after its local fan changes.
        for vertex in affected:
            for neighbor in links[vertex]:
                e=tuple(sorted((int(vertex),int(neighbor))))
                if e in blocked:
                    blocked.pop(e);queued.pop(e,None);new_edges.add(e)
        push_edges(sorted(new_edges))
        if count in goals:
            selected=np.flatnonzero(alive);remap=np.full(n,-1,np.int32);remap[selected]=np.arange(len(selected))
            if placement in ('edge','surface'):
                width=max(len(maps[int(i)]) for i in selected);indices=np.zeros((len(selected),width),np.int32);weights=np.zeros((len(selected),width))
                for row,i in enumerate(selected):
                    items=sorted(maps[int(i)].items(),key=lambda x:(-x[1],x[0]));indices[row,:len(items)]=[x[0] for x in items];weights[row,:len(items)]=[x[1] for x in items]
                outputs[count]={'indices':indices,'weights':weights,'offsets':np.zeros((len(selected),3)),'faces':remap[faces[f_alive]].copy(),'uv':uv[f_alive].copy(),'protected':ids[pinned].copy(),'graph':ids[graph].copy()}
            else:outputs[count]=(ids[selected].copy(),remap[faces[f_alive]].copy(),uv[f_alive].copy(),ids[pinned].copy(),ids[graph].copy())
            if on_checkpoint:on_checkpoint(count,outputs[count])
            print('ENDPOINT_CHECKPOINT',count,'body',int((~graph[alive]).sum()),'elapsed',round(time.monotonic()-start,2),flush=True)
        if accepted%2000==0:print('ENDPOINT_PROGRESS',count,'accepted',accepted,'rejected',rejected,'heap',len(heap),'seconds',round(time.monotonic()-start,1),flush=True)
    if count>target:print('ENDPOINT_EXHAUSTED',count,'target',target,flush=True)
    return outputs,{'target':target,'vertices':count,'accepted_collapses':accepted,'rejected_collapses':rejected,'frames':d,'seconds':time.monotonic()-start,'pinned_vertices':int(pinned.sum())}


def read_seed(folder):
    import bpy
    from scripts.character_lod.contract import require_hash
    folder=Path(folder);meta=json.loads((folder/'manifest.json').read_text())
    for name in ('model.blend','source-indices.npy','skin-uv.npy'):require_hash(folder/name,meta['files'][name])
    ids=np.load(folder/'source-indices.npy');uv=np.load(folder/'skin-uv.npy')
    with bpy.data.libraries.load(str(folder/'model.blend'),link=False) as (src,dst):dst.meshes=[meta['id']]
    data=dst.meshes[0];faces=np.array([p.vertices[:] for p in data.polygons],np.int32);bpy.data.meshes.remove(data)
    return ids,faces,uv,meta


def load_plan(path,seed,original_co,labels,hand_mask):
    from scripts.character_lod.correspondence import Correspondence
    from scripts.character_lod.contract import require_subset,APPROVED
    plan=np.load(path,allow_pickle=False)
    if 'offsets' in plan and np.any(plan['offsets']):raise ValueError('body surface plan requires zero offsets')
    if 'indices' in plan:mapping=Correspondence(plan['indices'],plan['weights'])
    else:
        ids=plan['source_indices'];mapping=Correspondence(ids[:,None],np.ones((len(ids),1)))
    faces=plan['faces'];uv=plan['uv'];protected=plan['protected'];graph=plan['graph'];n=len(mapping.indices)
    if n>=93038:raise ValueError('body target requires fewer than 93,038 vertices')
    if mapping.indices.max()>=len(original_co) or faces.min()<0 or faces.max()>=n:raise ValueError('plan indices exceed source')
    if uv.shape!=(len(faces),3,2) or not np.isfinite(uv).all():raise ValueError('invalid plan UV')
    if len(np.flatnonzero(mapping.exact_mask(protected)))!=len(protected):raise ValueError('protected point identity changed')
    if not set(np.flatnonzero(hand_mask)).issubset(graph):raise ValueError('full hand influence graph must stay frozen')
    seed_ids,seed_faces,_,meta=read_seed(seed)
    if meta['source_blend_sha256']!=APPROVED['male']['blend']:raise ValueError('unapproved frozen graph seed')
    fixed=seed_ids[~np.isin(labels[seed_ids],[3,4,5])]
    if not set(fixed).issubset(graph):raise ValueError('non-body graph must stay frozen')
    if int(mapping.exact_mask(fixed).sum())!=len(fixed):raise ValueError('seed non-body points changed')
    exact=(mapping.weights[:,0]==1)&(mapping.weights[:,1:]==0).all(1);surrogate=np.where(exact,mapping.indices[:,0],len(original_co)+np.arange(n))
    require_subset(surrogate,graph,seed_ids[seed_faces],faces)
    graph_rows=np.isin(surrogate,graph);touched=np.unique(faces[graph_rows[faces].any(1)])
    if not exact[touched].all():raise ValueError('frozen graph face gained an interpolated point')
    return mapping,faces,uv,protected,graph,meta


class SourceSurfaceProjector:
    """Constrain free body vertices to an actual original triangle in every frame."""
    def __init__(self,co,faces,uv,labels,hand_mask,normals,source_fields):
        from mathutils.bvhtree import BVHTree
        eligible=np.isin(labels[faces],[3,4,5]).all(1)&~hand_mask[faces].any(1)
        self.co=co;self.faces=faces[eligible];self.uv=uv.reshape(-1,3,2)[eligible];self.normals=normals;self.fields=source_fields
        self.tree=BVHTree.FromPolygons(co.tolist(),self.faces.tolist(),all_triangles=True)
    def __call__(self,point,first,second,t,expected_uv,uv_span):
        from mathutils import Vector
        normal=np.zeros(3)
        for vertex,weight in first.items():normal+=(1-t)*weight*self.normals[vertex]
        for vertex,weight in second.items():normal+=t*weight*self.normals[vertex]
        length=np.linalg.norm(normal)
        if length<1e-12:return None
        normal/=length;nearest=self.tree.find_nearest(Vector(point))
        if nearest[0] is None:return None
        candidates=[nearest]
        if np.dot(normal,nearest[1])<.3:candidates=self.tree.find_nearest_range(Vector(point),nearest[3]+.001)
        for location,face_normal,fi,distance in sorted(candidates,key=lambda x:x[3]):
            if np.dot(normal,face_normal)<.3:continue
            tri_ids=self.faces[fi];tri=self.co[tri_ids];a=tri[1]-tri[0];b=tri[2]-tri[0];p=np.asarray(location)-tri[0];aa=a@a;ab=a@b;bb=b@b;den=aa*bb-ab*ab
            if den<1e-25:continue
            w1=(bb*(p@a)-ab*(p@b))/den;w2=(aa*(p@b)-ab*(p@a))/den;weights=np.maximum([1-w1-w2,w1,w2],0);weights/=weights.sum();new_uv=weights@self.uv[fi]
            if np.linalg.norm(new_uv-expected_uv)>max(.02,2.5*uv_span):continue
            mapping={int(i):float(w) for i,w in zip(tri_ids,weights) if w>=1e-10};mass=sum(mapping.values());mapping={i:w/mass for i,w in mapping.items()}
            new_fields=np.einsum('i,dij->dj',weights,self.fields[:,tri_ids])
            return mapping,new_fields,new_uv
        return None
