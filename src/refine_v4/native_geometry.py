"""Native rest-geometry parts for an exactly corresponding canonical skin.

This adapter preserves the donor's axial roll and baked surface. Direction-only
retargeting cannot represent those. Different topology/weights, animated rigs,
or missing skin measurements are refused; there is no cross-rig approximation.
"""
import hashlib
import json
import numpy as np
from .pipeline import snapshot, skin_nonregression
from .relations import parse_leg_language
from .leg_adapter import measure_legs


def leg_name(name, sides=('left', 'right')):
    short = name.split(':')[-1]
    return any(short.startswith(s.title()+r) for s in sides
               for r in ('UpLeg', 'Leg', 'Foot', 'ToeBase'))


def _neutral(arm, meshes):
    if arm.animation_data or any(p.constraints for p in arm.pose.bones):
        raise ValueError('baked unconstrained native rest pose required')
    if any(np.max(abs(np.array(p.matrix_basis)-np.eye(4))) > 2e-5 for p in arm.pose.bones):
        raise ValueError('bake the current pose before native geometry transfer')
    if not meshes or any(m.data.shape_keys for m in meshes):
        raise ValueError('canonical meshes without shape keys required')
    if any(len(m.modifiers)!=1 or m.modifiers[0].type!='ARMATURE' or
           m.modifiers[0].object!=arm for m in meshes):
        raise ValueError('one matching armature modifier per mesh required')
    scale = arm.matrix_world.to_scale()
    if min(scale)<=0 or max(scale)-min(scale)>1e-5:
        raise ValueError('uniform positive armature scale required')


def _weights(mesh):
    names={g.index:g.name for g in mesh.vertex_groups}
    return [{names[g.group]:float(g.weight) for g in v.groups} for v in mesh.data.vertices]


def _digest(part):
    return hashlib.sha256(json.dumps(part,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _check_relation(arm, intent):
    relation=parse_leg_language(intent.get('sentence',''))
    top=intent.get('top_side') or relation.get('top_side')
    if relation.get('crossing') or top:
        if intent.get('cross_region')!='knee' or top not in ('left','right'):
            raise ValueError('native crossing requires a checked knee family and explicit top side')
        measured=measure_legs(arm)
        if measured['top_side']!=top or measured['knee_flex_deg'][top]<45:
            raise ValueError('native anatomical crossing order does not match declared intent')
        return measured
    return None


def extract_geometry_part(arm, meshes, *, sides, source_sha256, intent):
    """Capture an authored native part; this never grants automatic approval."""
    _neutral(arm, meshes)
    if not sides or len(set(sides))!=len(sides) or not set(sides)<={'left','right'}:
        raise ValueError('explicit unique anatomical sides required')
    if len(source_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('source SHA256 required')
    if not intent.get('reference') or intent.get('source') not in ('user_confirmed','reviewed_library'):
        raise ValueError('attributed native part intent required')
    measured=_check_relation(arm,intent)
    hip=next(p for p in arm.pose.bones if p.name.split(':')[-1]=='Hips')
    pelvis=arm.matrix_world@hip.matrix;inverse=pelvis.inverted()
    part={'schema':'standin.refine-v4.native-geometry-part.v1','sides':list(sides),
          'source_sha256':source_sha256,'intent':intent,'measured_relation':measured,'joints':{},'lengths':{},'meshes':[]}
    for p in arm.pose.bones:
        if leg_name(p.name,sides):
            part['joints'][p.name]=np.array(inverse@arm.matrix_world@p.matrix).tolist()
            part['lengths'][p.name]=float(p.length)
    for m in meshes:
        part['meshes'].append({'topology':[list(p.vertices) for p in m.data.polygons],
          'weights':_weights(m),
          'vertices_pelvis':[list(inverse@m.matrix_world@v.co) for v in m.data.vertices]})
    part['id']=_digest(part)
    return part


def apply_geometry_part(arm, meshes, part, *, skin_check):
    """Atomically replace selected rest skin/frames, retaining root and upper body."""
    import bpy
    from mathutils import Matrix,Vector
    _neutral(arm,meshes)
    if not callable(skin_check):raise ValueError('actual skin checker required')
    if part.get('id')!=_digest({k:v for k,v in part.items() if k!='id'}):
        raise ValueError('native part fingerprint mismatch')
    sides=part['sides']
    if not sides or len(set(sides))!=len(sides) or not set(sides)<={'left','right'}:
        raise ValueError('invalid native part sides')
    if len(meshes)!=len(part['meshes']):raise ValueError('canonical mesh count mismatch')
    expected={p.name for p in arm.pose.bones if leg_name(p.name,sides)}
    if set(part['joints'])!=expected or set(part['lengths'])!=expected:
        raise ValueError('named leg chain mismatch')
    for p in arm.pose.bones:
        if p.name in expected and abs(p.length-part['lengths'][p.name])>2e-5:
            raise ValueError('canonical bone lengths mismatch')
    for m,src in zip(meshes,part['meshes']):
        if [list(p.vertices) for p in m.data.polygons]!=src['topology'] or len(m.data.vertices)!=len(src['vertices_pelvis']):
            raise ValueError('canonical topology mismatch')
        for w,sw in zip(_weights(m),src['weights']):
            if max((abs(w.get(k,0)-sw.get(k,0)) for k in set(w)|set(sw)),default=0)>1e-5:
                raise ValueError('canonical named weights mismatch')
    before=snapshot(arm);baseline=skin_check(arm)
    rest=arm.data.copy();saved=[m.data.copy() for m in meshes]
    result={'accepted':False,'status':'not_applied','method':'native_rest_geometry_part',
            'part_id':part['id'],'reasons':[],'release_allowed':False}
    try:
        if not skin_nonregression(baseline,baseline)['accepted']:
            raise ValueError('complete baseline skin measurement required')
        hip=next(p for p in arm.pose.bones if p.name.split(':')[-1]=='Hips')
        pelvis=arm.matrix_world@hip.matrix
        copied=0
        for m,src in zip(meshes,part['meshes']):
            for v,w,co in zip(m.data.vertices,_weights(m),src['vertices_pelvis']):
                total=sum(value for n,value in w.items() if leg_name(n,sides))
                x=np.clip((total-.03)/.47,0,1);alpha=float(x*x*(3-2*x))
                v.co=v.co.lerp(m.matrix_world.inverted()@pelvis@Vector(co),alpha)
                copied+=alpha==1
            m.data.update()
        lengths={p.name:p.length for p in arm.data.bones}
        bpy.ops.object.select_all(action='DESELECT');arm.select_set(True)
        bpy.context.view_layer.objects.active=arm;bpy.ops.object.mode_set(mode='EDIT')
        for p in arm.data.edit_bones:
            if p.name in expected:
                p.matrix=arm.matrix_world.inverted()@pelvis@Matrix(part['joints'][p.name]);p.length=lengths[p.name]
        bpy.ops.object.mode_set(mode='OBJECT');bpy.context.view_layer.update()
        after=snapshot(arm)
        protected=max(float(np.max(abs(after[n]-v))) for n,v in before.items() if n not in expected)
        template=max(float(np.max(abs(after[n]-np.array(pelvis)@np.array(v)))) for n,v in part['joints'].items())
        if protected>2e-5 or template>2e-5:raise ValueError('native part joint preservation failed')
        result['measured_relation']=_check_relation(arm,part['intent'])
        check=skin_nonregression(baseline,skin_check(arm));result['skin_check']=check
        if not check['accepted']:raise ValueError('native part skin rejected')
        result.update(accepted=True,status='applied',protected_matrix_delta=protected,
                      template_matrix_delta=template,exact_lower_vertices=copied)
    except Exception as exc:
        if arm.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        arm.data=rest
        for m,data in zip(meshes,saved):m.data=data
        bpy.context.view_layer.update()
        result['reasons'].append(str(exc))
        result['rollback_matrix_delta']=max(float(np.max(abs(snapshot(arm)[n]-v))) for n,v in before.items())
    finally:
        if rest.users==0:bpy.data.armatures.remove(rest)
        for data in saved:
            if data.users==0:bpy.data.meshes.remove(data)
    return result


def advance_foot(arm, *, side, distance_m):
    """Pose-only fixed-length IK. Use inside run_refine_v4 with final skin checks."""
    import bpy
    from mathutils import Matrix
    from src.experimental.part_composition import shortest_arc
    if side not in ('left','right') or not np.isfinite(distance_m) or not 0<distance_m<=.1:
        raise ValueError('anatomical foot side and forward distance in (0, .1] required')
    b={p.name.split(':')[-1]:p for p in arm.pose.bones};s=side.title()
    hip,knee,foot=[b[s+r] for r in ('UpLeg','Leg','Foot')]
    if knee.parent!=hip or foot.parent!=knee:raise ValueError('unsupported foot chain')
    R=(arm.matrix_world@b['Hips'].matrix).to_3x3().normalized()
    forward=np.array(R.col[2]);forward[2]=0;forward/=np.linalg.norm(forward)
    if not np.isfinite(forward).all():raise ValueError('horizontal anterior direction undefined')
    toe=np.array(arm.matrix_world@b[s+'ToeBase'].head-arm.matrix_world@foot.head)
    if np.dot(toe,forward)<=0:raise ValueError('anterior direction disagrees with current toes')
    h,k,f=[np.array(arm.matrix_world@p.head) for p in (hip,knee,foot)]
    lengths=[np.linalg.norm(k-h),np.linalg.norm(f-k)];orientation=(arm.matrix_world@foot.matrix).to_3x3().copy()
    target=f+forward*distance_m;delta=target-h;d=np.linalg.norm(delta)
    if not abs(lengths[0]-lengths[1])+1e-6<d<sum(lengths)-1e-6:raise ValueError('forward foot target unreachable')
    u=delta/d;pole=k-h-u*np.dot(k-h,u)
    if np.linalg.norm(pole)<1e-6:raise ValueError('current knee pole undefined')
    pole/=np.linalg.norm(pole);a=(lengths[0]**2-lengths[1]**2+d*d)/(2*d)
    goal=h+u*a+pole*np.sqrt(lengths[0]**2-a*a)
    for parent,child,direction in ((hip,knee,goal-h),(knee,foot,target-goal)):
        ray=np.array(arm.matrix_world@child.head-arm.matrix_world@parent.head)
        world=arm.matrix_world@parent.matrix;rotation=Matrix(shortest_arc(ray,direction).tolist())
        desired=(rotation@world.to_3x3()).to_4x4();desired.translation=world.translation
        parent.matrix=arm.matrix_world.inverted()@desired;bpy.context.view_layer.update()
    desired=orientation.to_4x4();desired.translation=(arm.matrix_world@foot.matrix).translation
    foot.matrix=arm.matrix_world.inverted()@desired;bpy.context.view_layer.update()
    actual=np.array(arm.matrix_world@foot.head)-f
    if abs(float(actual@forward)-distance_m)>1e-5 or abs(actual[2])>1e-5:
        raise ValueError('saved foot target not reached')
    return {'accepted':True,'reasons':[],'method':'fixed_length_anterior_foot_ik',
            'advance_m':float(actual@forward),'ankle_height_delta_m':float(actual[2])}
