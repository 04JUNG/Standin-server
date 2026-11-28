"""Rebuild selected arm skin from the matching canonical mesh, once per pose.

Repeated rest-pose LBS baking can accumulate candy-wrapper collapse. Exact
topology correspondence permits the original surface to be deformed with rigid
dual-quaternion blends instead. This is opt-in, never nearest-neighbour transfer,
and never modifies bones, topology, weights, or vertices outside the arm mask.
"""
import numpy as np
import bpy
from mathutils import Matrix,Vector
from .contact_adapter import bones

def _weights(obj):
    names={g.index:g.name.split(':')[-1] for g in obj.vertex_groups}
    return [[(names[g.group],g.weight) for g in v.groups if g.weight>0] for v in obj.data.vertices]

def capture_canonical(arm,meshes):
    if len(meshes)!=1:raise ValueError('single canonical mesh required')
    obj=meshes[0]
    return {'vertices':np.array([obj.matrix_world@v.co for v in obj.data.vertices]),
            'faces':[tuple(p.vertices) for p in obj.data.polygons],'weights':_weights(obj),
            'bones':{n:np.array(arm.matrix_world@p.matrix) for n,p in bones(arm).items()}}

def _multiply(a,b):
    aw,av=a[...,0:1],a[...,1:];bw,bv=b[...,0:1],b[...,1:]
    return np.concatenate((aw*bw-np.sum(av*bv,axis=-1,keepdims=True),aw*bv+bw*av+np.cross(av,bv)),axis=-1)

def _thickness(vertices,arm,side):
    b=bones(arm);s,e,w=[np.array(arm.matrix_world@b[side+t].head) for t in ['Arm','ForeArm','Hand']]
    normal=np.cross(e-s,w-e);length=np.linalg.norm(normal)
    if length<1e-8:return None
    nearby=vertices[np.linalg.norm(vertices-e,axis=1)<.035]
    if len(nearby)<8:return None
    projected=(nearby-e)@(normal/length)
    return float(np.quantile(projected,.95)-np.quantile(projected,.05))

def rebuild_selected_surface(arm,meshes,canonical,sides):
    if len(meshes)!=1 or not sides or not set(sides)<={'left','right'}:raise ValueError('explicit sides and single mesh required')
    obj=meshes[0];data=obj.data
    if len(data.vertices)!=len(canonical['vertices']) or [tuple(p.vertices) for p in data.polygons]!=canonical['faces']:
        raise ValueError('exact canonical topology correspondence required')
    if any(max(abs(p.matrix_basis[r][c]-(1. if r==c else 0.)) for r in range(4) for c in range(4))>1e-5 for p in arm.pose.bones):
        raise ValueError('bake the selected pose before canonical surface rebuild')
    current=np.array([obj.matrix_world@v.co for v in data.vertices]);weights=_weights(obj)
    prefixes=[s.title() for s in sides]
    selected=np.array([sum(v for n,v in ws if any(n in [s+'Arm',s+'ForeArm'] or n.startswith(s+'Hand') for s in prefixes)) for ws in weights])
    mask=selected>.05;indices=np.flatnonzero(mask)
    b=bones(arm);transforms={}
    for name,source in canonical['bones'].items():
        if name not in b:raise ValueError('canonical bone missing: '+name)
        target=np.array(arm.matrix_world@b[name].matrix)
        rotation=Matrix(target[:3,:3]).normalized()@Matrix(source[:3,:3]).normalized().transposed()
        q=np.array(rotation.to_quaternion());translation=target[:3,3]-np.array(rotation)@source[:3,3]
        d=.5*_multiply(np.r_[0.,translation],q);transforms[name]=(q,d)
    real=[];dual=[]
    for i in indices:
        ws=canonical['weights'][i]
        if not ws:raise ValueError('unweighted canonical vertex')
        reference=transforms[max(ws,key=lambda v:v[1])[0]][0]
        qr=np.zeros(4);qd=np.zeros(4)
        for name,weight in ws:
            q,d=transforms[name];sign=1 if q@reference>=0 else -1;qr+=q*weight*sign;qd+=d*weight*sign
        length=np.linalg.norm(qr)
        if length<1e-8:raise ValueError('degenerate dual-quaternion blend')
        qr/=length;qd/=length;qd-=qr*(qr@qd);real.append(qr);dual.append(qd)
    real=np.array(real);dual=np.array(dual);v=canonical['vertices'][indices]
    xyz=real[:,1:];rotated=v+2*np.cross(xyz,np.cross(xyz,v)+real[:,0:1]*v)
    conjugate=real.copy();conjugate[:,1:]*=-1
    posed=rotated+2*_multiply(dual,conjugate)[:,1:]
    # Preserve the shoulder boundary with a smooth arm-weight mask.
    factor=np.clip((selected[indices]-.05)/.90,0,1);factor=factor*factor*(3-2*factor)
    proposed=current.copy();proposed[indices]=current[indices]+(posed-current[indices])*factor[:,None]
    if not np.isfinite(proposed).all():raise ValueError('nonfinite canonical surface result')
    old={s:_thickness(current,arm,s.title()) for s in sides}
    inverse=obj.matrix_world.inverted()
    for i in indices:data.vertices[int(i)].co=inverse@Vector(proposed[i])
    data.update();bpy.context.view_layer.update()
    protected=float(np.max(abs(proposed[~mask]-current[~mask]))) if np.any(~mask) else 0.
    assert protected==0
    return {'method':'exact_topology_canonical_dual_quaternion_arm_surface','selected_sides':list(sides),
            'changed_vertices':len(indices),'nonselected_surface_max_delta_m':protected,
            'max_selected_surface_move_m':float(np.max(np.linalg.norm(proposed-current,axis=1))),
            'elbow_thickness_before_m':old,'elbow_thickness_after_m':{s:_thickness(proposed,arm,s.title()) for s in sides},
            'topology_changed':False,'weights_changed':False,'bones_changed':False}
