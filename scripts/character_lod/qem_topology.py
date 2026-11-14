"""QEM interior mesh with exact protected graph and frozen source projection."""
import bpy,numpy as np
from scripts.character_lod.contract import require_subset
from scripts.character_lod.correspondence import Correspondence
from scripts.character_upgrade.build_source import projection


def simplify(mesh,co,protected,source_uv,ratio,keep=None):
    source_uv=np.asarray(source_uv).reshape(-1,3,2)
    original_faces=np.array([p.vertices[:] for p in mesh.data.polygons],np.int32)
    edges=np.array([e.vertices[:] for e in mesh.data.edges]);frozen=protected.copy()
    frozen[edges[np.any(protected[edges],axis=1)].ravel()]=True
    if keep is not None:
        frozen|=keep
    n=mesh.copy();n.data=mesh.data.copy();bpy.context.collection.objects.link(n)
    n['standin_studio_mesh']=False;n.shape_key_clear();n.modifiers.clear();n.parent=None
    n.data.vertices.foreach_set('co',co.ravel())
    layer=n.data.uv_layers.new(name='source_uv')
    for loop,value in zip(layer.data,source_uv.reshape(-1,2)):loop.uv=value
    attr=n.data.attributes.new('original_id','INT','POINT');attr.data.foreach_set('value',np.arange(len(co),dtype=np.int32))
    group=n.vertex_groups.new(name='qem_interior');group.add(np.flatnonzero(~frozen).tolist(),1.,'REPLACE')
    try:
        bpy.ops.object.select_all(action='DESELECT');n.select_set(True);bpy.context.view_layer.objects.active=n
        mod=n.modifiers.new('QEM interior','DECIMATE');mod.ratio=ratio;mod.vertex_group=group.name;mod.vertex_group_factor=1000.;mod.use_collapse_triangulate=True
        bpy.ops.object.modifier_apply(modifier=mod.name)
        points=np.array([v.co[:] for v in n.data.vertices]);faces=np.array([p.vertices[:] for p in n.data.polygons],np.int32)
        uv=np.array([l.uv[:] for l in n.data.uv_layers.active.data]).reshape(-1,3,2)
        hints=np.array([v.value for v in n.data.attributes['original_id'].data])
        exact=np.linalg.norm(points-co[hints],axis=1)<1e-7
        # New points receive unique surrogate IDs only for the graph guard.
        graph_ids=np.where(exact,hints,len(co)+np.arange(len(points)))
        require_subset(graph_ids,np.flatnonzero(protected),original_faces,faces)
        if keep is not None and not set(np.flatnonzero(keep)).issubset(graph_ids):
            raise ValueError('QEM changed a required deformation support point')
        indices=np.repeat(hints[:,None],3,axis=1).astype(np.int32);weights=np.zeros((len(points),3));weights[:,0]=1.
        unknown=np.flatnonzero(~exact)
        pi,pw,distance=projection(points[unknown],co,original_faces)
        if (pw < -1e-4).any():raise ValueError('projection left source triangle')
        pw=np.maximum(pw,0);pw/=pw.sum(1)[:,None];indices[unknown]=pi;weights[unknown]=pw
        offsets=np.zeros((len(points),3))
        tri=co[pi];u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0];cross=np.cross(u,v)
        normal=cross/np.sqrt(np.linalg.norm(cross,axis=1))[:,None]
        origin=np.einsum('ni,nij->nj',pw,tri);frame=np.stack((u,v,normal),axis=2)
        offsets[unknown]=np.linalg.solve(frame,(points[unknown]-origin)[...,None])[...,0]
        mapping=Correspondence(indices,weights,offsets)
        projected=mapping.apply(co)
        before=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
        after=np.cross(projected[faces[:,1]]-projected[faces[:,0]],projected[faces[:,2]]-projected[faces[:,0]])
        if np.any(np.einsum('ij,ij->i',before,after)<=0) or np.any(np.linalg.norm(after,axis=1)<2e-12):
            raise ValueError('source projection inverted or collapsed a QEM triangle')
        # Protected and unchanged original faces use their exact source UVs.
        lookup={tuple(sorted(f)):i for i,f in enumerate(original_faces)}
        for i,face in enumerate(faces):
            key=tuple(sorted(graph_ids[face]));old=lookup.get(key)
            if old is not None:
                positions={int(sid):j for j,sid in enumerate(original_faces[old])}
                uv[i]=[source_uv[old,positions[int(sid)]] for sid in graph_ids[face]]
        return projected,faces,uv,mapping,{'ratio':ratio,'new_projected_vertices':len(unknown),'projection_max_mm':float(distance.max()*1000) if len(distance) else 0.,'protected_graph_verified':True,'correspondence':'barycentric plus fixed source-triangle frame offset'}
    finally:
        bpy.data.objects.remove(n,do_unlink=True)
