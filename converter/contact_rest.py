"""Opt-in local head/forearm contact solve on an already retargeted rig.

Only called when image inference selects a head-on-forearms surface-rest class.
All targets are derived from the character's current geometry. No cut IDs,
person-specific coordinates, pose IDs or desired angles are accepted.
"""
from __future__ import annotations
import math
import numpy as np


def solve_head_forearm_rest(arm,meshes):
    import bpy
    from mathutils import Matrix,Vector
    from mathutils.kdtree import KDTree
    from . import retarget as rt
    names=('Spine1','Spine2','Neck','Head')
    bones={name:arm.pose.bones['mixamorig:'+name] for name in names}
    rests={name:pb.matrix_basis.copy() for name,pb in bones.items()}
    def joint(name):return arm.matrix_world @ arm.pose.bones['mixamorig:'+name].head
    right=joint('LeftArm')-joint('RightArm');right.z=0;right.normalize()
    up=Vector((0,0,1));head= bones['Head']
    # Head skin landmarks in the current head-bone frame.
    vertices=[]
    for mesh in meshes:
        group=mesh.vertex_groups.get('mixamorig:Head')
        if group is None:continue
        for vertex in mesh.data.vertices:
            if any(g.group==group.index and g.weight>=.55 for g in vertex.groups):vertices.append(mesh.matrix_world @ vertex.co)
    if len(vertices)<20:raise ValueError('head contact solve requires weighted head skin')
    head_frame=arm.matrix_world @ head.matrix
    local=[head_frame.inverted() @ p for p in vertices]
    center=sum(vertices,Vector())/len(vertices)
    center_local=head_frame.inverted() @ center
    # Bone-level objective uses skin radius; post-solve surface measurement is
    # saved independently below. The radius is derived from this exact mesh.
    radius=float(np.quantile([abs((p-center).dot(up)) for p in vertices],.80))
    original={name:joint(name) for name in ('LeftForeArm','LeftHand','RightForeArm','RightHand')}
    targets=[];constraints=[]
    for side in ('Left','Right'):
        bpy.ops.object.empty_add(location=original[side+'Hand']);target=bpy.context.object;targets.append(target)
        pb=arm.pose.bones['mixamorig:'+side+'ForeArm'];constraint=pb.constraints.new('IK');constraint.target=target;constraint.chain_count=2;constraint.use_stretch=False;constraint.iterations=100
        constraints.append((pb,constraint))
    def rotate(pb,degrees):
        pivot=arm.matrix_world @ pb.head
        world=arm.matrix_world @ pb.matrix
        transform=Matrix.Translation(pivot) @ Matrix.Rotation(math.radians(degrees),4,right) @ Matrix.Translation(-pivot)
        pb.matrix=arm.matrix_world.inverted() @ transform @ world
        bpy.context.view_layer.update()
    def apply(values):
        for name,pb in bones.items():pb.matrix_basis=rests[name].copy()
        for side,target in zip(('Left','Right'),targets):target.location=original[side+'Hand']+up*values[4]
        bpy.context.view_layer.update()
        for name,value in zip(names,values[:4]):rotate(bones[name],value)
        bpy.context.view_layer.update()
    def measure():
        h=arm.matrix_world @ head.matrix @ center_local
        arms=[(joint(s+'ForeArm'),joint(s+'Hand')) for s in ('Left','Right')]
        support=sum((a+b for a,b in arms),Vector())/4
        delta=h-support
        vertical=delta.dot(up)-radius-.035
        horizontal=(delta-up*delta.dot(up)).length
        forearm_vertical=np.mean([abs((b-a).dot(up))/max((b-a).length,1e-6) for a,b in arms])
        return dict(vertical_gap_m=float(vertical),horizontal_offset_m=float(horizontal),forearm_vertical_fraction=float(forearm_vertical),head_center=list(h),support_center=list(support))
    def score(values):
        apply(values);m=measure()
        objective=7*m['vertical_gap_m']**2+3*m['horizontal_offset_m']**2+.015*m['forearm_vertical_fraction']**2
        objective+=.0000007*sum(v*v for v in values[:4])+.05*values[4]**2
        # Keep crossed forearms rather than solving contact with a raised palm.
        if m['vertical_gap_m']<-.015:objective+=30*(m['vertical_gap_m']+.015)**2
        return float(objective),m
    start=[0.,0.,0.,0.,0.];before=score(start)[1]
    best=start[:];best_score,_=score(best);evaluations=1
    bounds=[(-35.,35.),(-35.,35.),(-35.,35.),(-35.,35.),(-.03,.15)]
    for step,arm_step in ((15.,.06),(7.5,.03),(3.,.012),(1.,.004)):
        for sweep in range(3):
            changed=False
            for axis in range(5):
                base=best[:];increment=arm_step if axis==4 else step
                for sign in (-1,1):
                    candidate=base[:];candidate[axis]=max(bounds[axis][0],min(bounds[axis][1],candidate[axis]+sign*increment))
                    value,_=score(candidate);evaluations+=1
                    if value<best_score:best_score=value;best=candidate;changed=True
            if not changed:break
    apply(best);after=measure()
    # Bake the evaluated IK rotations before removing helper constraints.
    matrices={pb.name:pb.matrix.copy() for pb in arm.pose.bones}
    for pb,c in constraints:pb.constraints.remove(c)
    for target in targets:bpy.data.objects.remove(target,do_unlink=True)
    for pb in arm.pose.bones:pb.matrix=matrices[pb.name]
    bpy.context.view_layer.update()
    after_bake=measure()
    if after_bake['vertical_gap_m']>.04 or after_bake['horizontal_offset_m']>.12:raise ValueError('automatic forearm contact solve did not converge: '+str(after_bake))
    original_lengths={bone.name:bone.length for bone in arm.data.bones}
    rt.apply_output_mode(arm,meshes,'rigged_rest')
    head_skin=[];forearm_skin=[]
    for mesh in meshes:
        group_names={g.index:g.name.split(':')[-1] for g in mesh.vertex_groups}
        for vertex in mesh.data.vertices:
            weights={group_names.get(g.group,''):g.weight for g in vertex.groups}
            point=mesh.matrix_world @ vertex.co
            if weights.get('Head',0)>.55:head_skin.append(point)
            if sum(weights.get(name,0) for name in ('LeftForeArm','RightForeArm'))>.55:forearm_skin.append(point)
    if not head_skin or not forearm_skin:raise ValueError('contact skin groups missing')
    tree=KDTree(len(forearm_skin))
    for index,point in enumerate(forearm_skin):tree.insert(point,index)
    tree.balance();minimum=min(tree.find(point)[2] for point in head_skin)
    length_error=max(abs(bone.length-original_lengths[bone.name]) for bone in arm.data.bones)
    if length_error>1e-5:raise ValueError('contact solve changed bone lengths')
    return dict(status='automatic_geometry_contact_solve',before=before,after=after_bake,head_forearm_skin_vertex_clearance_m=minimum,max_bone_length_change_m=length_error,rotation_deltas_degrees=dict(zip(names,best[:4])),hand_lift_m=best[4],head_skin_radius_m=radius,evaluations=evaluations,manual_targets=False)
