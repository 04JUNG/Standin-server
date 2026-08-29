#!/usr/bin/env python3
"""Independent CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY verifier (Blender, QA only).

Loads only bone-map data.  It does not import converter.retarget or reuse solver helpers.
It independently rebuilds the Hips-seeded leg transport and frozen V3.1 ankle policy,
checks source-pose edge directions, and checks target-rest-relative terminal frames.
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, math, os

import bpy
from mathutils import Matrix, Vector


def args_parse():
    av = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--map-root", required=True)
    p.add_argument("--bvh", required=True)
    p.add_argument("--character", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--mirror", choices=("true", "false"), default="false")
    return p.parse_args(av)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_bvh(path, frame):
    reset(); before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=path, axis_forward="-Z", axis_up="Y",
                            rotate_mode="NATIVE", use_fps_scale=False,
                            update_scene_fps=False, update_scene_duration=True)
    arm = [o for o in bpy.data.objects if o not in before and o.type == "ARMATURE"][0]
    sc = bpy.context.scene
    target = max(sc.frame_start, min(sc.frame_start + frame, sc.frame_end))
    sc.frame_set(target)
    bpy.ops.object.select_all(action="DESELECT"); arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    sc.frame_set(target); bpy.context.view_layer.update()
    return arm


def import_fbx(path):
    reset(); before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=True,
                             automatic_bone_orientation=False)
    arm = [o for o in bpy.data.objects if o not in before and o.type == "ARMATURE"][0]
    bpy.context.view_layer.update(); return arm


def resolve(names, profiles, canon):
    s = set(names); rows=[]
    for p,t in profiles.items():
        if t: rows.append((sum(t.get(c) in s for c in canon), p))
    return max(rows)[1]


def reflect(v):
    return Vector((-v.x, v.y, v.z))


def angle(a,b,unsigned=False):
    if a.length < 1e-10 or b.length < 1e-10: return None
    d=a.normalized().dot(b.normalized())
    if unsigned: d=abs(d)
    return math.degrees(math.acos(max(-1.0,min(1.0,d))))


def qerr(a,b):
    d=abs(a.normalized().dot(b.normalized()))
    return math.degrees(2*math.acos(max(-1.0,min(1.0,d))))


def signed_twist(rotation, axis):
    if axis.length < 1e-10: return None
    u=axis.normalized(); q=rotation.to_quaternion().normalized()
    projected=q.x*u.x+q.y*u.y+q.z*u.z
    value=2*math.atan2(projected,q.w)
    while value>math.pi: value-=2*math.pi
    while value<-math.pi: value+=2*math.pi
    return math.degrees(value)


def minrot(a,b):
    if a.length < 1e-10 or b.length < 1e-10: return None
    u,v=a.normalized(),b.normalized(); d=max(-1.0,min(1.0,u.dot(v)))
    ang=math.acos(d)
    if ang < math.radians(.5): return Matrix.Identity(3)
    if ang > math.radians(175): return None
    axis=u.cross(v)
    if axis.length < 1e-10: return None
    return Matrix.Rotation(ang,3,axis.normalized())


def scaled_minrot(full, mu):
    if mu <= 0.0: return Matrix.Identity(3)
    if mu >= 1.0: return full.copy()
    axis,ang=full.to_quaternion().normalized().to_axis_angle()
    return Matrix.Rotation(ang*mu,3,axis)


def frame(direction, normal):
    y=direction.normalized(); z=normal-y*normal.dot(y)
    if z.length < 1e-10: return None
    z.normalize(); x=y.cross(z).normalized(); z=x.cross(y).normalized()
    return Matrix(((x.x,y.x,z.x),(x.y,y.y,z.y),(x.z,y.z,z.z)))


def role_bones(arm, hand):
    q=[(c,1) for c in arm.data.bones[hand].children]; rows={k:[] for k in ('index','pinky','thumb')}
    while q:
        b,depth=q.pop(0); low=b.name.lower(); role=None; explicit=0
        if 'thumb' in low: role,explicit='thumb',1
        elif 'index' in low: role,explicit='index',1
        elif 'pinky' in low or 'little' in low: role,explicit='pinky',1
        elif 'fingerbase' in low: role='index'
        if role: rows[role].append((-explicit,depth,b.name))
        q.extend((c,depth+1) for c in b.children)
    return {k:sorted(v)[0][2] for k,v in rows.items() if v}


def palm(arm, hand, pose, roles=None, mirrored=False):
    rb=role_bones(arm,hand)
    if roles is None:
        roles=('index','pinky') if all(x in rb for x in ('index','pinky')) else \
              ('index','thumb') if all(x in rb for x in ('index','thumb')) else None
    if roles is None or any(x not in rb for x in roles): return None,roles,rb
    if pose:
        o=arm.matrix_world@arm.pose.bones[hand].head
        pts=[arm.matrix_world@arm.pose.bones[rb[x]].tail for x in roles]
    else:
        o=arm.matrix_world@arm.data.bones[hand].head_local
        pts=[arm.matrix_world@arm.data.bones[rb[x]].tail_local for x in roles]
    if mirrored: o=reflect(o); pts=[reflect(x) for x in pts]
    rays=[x-o for x in pts]
    if min(x.length for x in rays)<1e-10: return None,roles,rb
    u,v=[x.normalized() for x in rays]; n=u.cross(v); f=u+v
    if n.length<1e-6 or f.length<1e-6: return None,roles,rb
    return frame(f,n),roles,rb


def extract(arm, tbl, canon, pose, canonical_lookup=None, mirrored=False):
    pts={}; rots={}
    for c in canon:
        sc=canonical_lookup(c) if canonical_lookup else c
        n=tbl.get(sc)
        if not n or n not in arm.data.bones: continue
        if pose:
            m=arm.matrix_world@arm.pose.bones[n].matrix
        else:
            m=arm.matrix_world@arm.data.bones[n].matrix_local
        p=m.translation.copy(); q=m.to_quaternion()
        if mirrored:
            p=reflect(p)
            mm=Matrix.Diagonal((-1,1,1,1))@q.to_matrix().to_4x4()@Matrix.Diagonal((-1,1,1,1))
            q=mm.to_quaternion()
        pts[c]=p; rots[c]=q
    return pts,rots


def main():
    a=args_parse(); mirror=a.mirror=='true'
    root=os.path.abspath(a.map_root); sys.path.insert(0,root)
    from converter.bone_map import CANONICAL_BONES, PROFILES, mirror_name
    if any('converter.retarget' == x or x.endswith('.retarget') for x in sys.modules):
        raise SystemExit('[FAIL] converter.retarget loaded')
    rep=json.load(open(a.report,encoding='utf-8'))

    sa=import_bvh(a.bvh,a.frame); sp=resolve([b.name for b in sa.data.bones],PROFILES,CANONICAL_BONES); st=PROFILES[sp]
    lookup=mirror_name if mirror else None
    spts,srot=extract(sa,st,CANONICAL_BONES,True,lookup,mirror)
    srpts,srrot=extract(sa,st,CANONICAL_BONES,False,lookup,mirror)
    spalm={}
    for side in ('L','R'):
        sc=mirror_name('hand.'+side) if mirror else 'hand.'+side
        hand=st[sc]; spalm[side]=palm(sa,hand,True,mirrored=mirror)

    ta=import_fbx(a.character); tp=resolve([b.name for b in ta.data.bones],PROFILES,CANONICAL_BONES); tt=PROFILES[tp]
    tpts,trot=extract(ta,tt,CANONICAL_BONES,False)
    tpalm={side:palm(ta,tt['hand.'+side],False,roles=spalm[side][1]) for side in ('L','R')}

    oa=import_fbx(a.artifact); op=resolve([b.name for b in oa.data.bones],PROFILES,CANONICAL_BONES); ot=PROFILES[op]
    opts,orot=extract(oa,ot,CANONICAL_BONES,False)
    opalm={side:palm(oa,ot['hand.'+side],False,roles=spalm[side][1]) for side in ('L','R')}

    chains={
      'arm.L':('upperarm.L','forearm.L','hand.L'), 'arm.R':('upperarm.R','forearm.R','hand.R'),
      'leg.L':('upleg.L','leg.L','foot.L','toe.L'), 'leg.R':('upleg.R','leg.R','foot.R','toe.R')}
    expected_hips=(srot['hips'].to_matrix()@srrot['hips'].to_matrix().transposed()
                   @trot['hips'].to_matrix())
    hips_error=qerr(expected_hips.to_quaternion(),orot['hips'])
    hips_transport=orot['hips'].to_matrix()@trot['hips'].to_matrix().transposed()
    out={'ok':True,'profiles':{'source':sp,'target':tp,'artifact':op},'chains':{},
         'terminal_relative':{},
         'pelvis_boundary':{'hips_legacy_rotation_error_deg':hips_error,
                            'hips_transport_determinant':hips_transport.determinant(),
                            'pass':hips_error<=0.2 and hips_transport.determinant()>0.999},
         'gates':{'direction_deg':0.2,'rotation_deg':0.2,'terminal_relative_deg':1.0},
         'production_retarget_loaded':False}
    out['ok'] &= out['pelvis_boundary']['pass']
    for name,nodes in chains.items():
        first_mode=rep.get('solver_mode_by_bone',{}).get(nodes[0],'')
        if first_mode in ('legacy_compatible_chain','chain_degenerate_fallback'):
            errs=[]
            for c in nodes:
                expected=srot[c]@srrot[c].inverted()@trot[c]
                errs.append(qerr(expected,orot[c]))
            row={'mode':first_mode,'legacy_rotation_error_deg':errs,
                 'pass':all(x<=1.0 for x in errs)}
            out['chains'][name]=row; out['ok'] &= row['pass']; continue
        parent_coherent=(first_mode=='chain_transport_parent_coherent')
        if first_mode not in ('chain_transport','chain_transport_parent_coherent'):
            row={'mode':first_mode,'pass':False,'reason':'unexpected active mode'}
            out['chains'][name]=row; out['ok']=False; continue
        foot_mode=rep.get('solver_mode_by_bone',{}).get(nodes[2])
        count=(2 if name.startswith('arm') else
               3 if foot_mode in ('chain_transport','chain_transport_partial',
                                  'chain_transport_parent_coherent',
                                  'chain_transport_parent_coherent_partial') else 2)
        se=[(spts[nodes[i+1]]-spts[nodes[i]]).normalized() for i in range(count)]
        de=[(tpts[nodes[i+1]]-tpts[nodes[i]]).normalized() for i in range(count)]
        oe=[opts[nodes[i+1]]-opts[nodes[i]] for i in range(count)]
        q=(hips_transport.copy() if parent_coherent else Matrix.Identity(3))
        rotation_errors=[]; increment_deg=[]; expected_edges=[]; failed=False
        q_first=None
        for i in range(count):
            predicted=q@de[i]; increment_deg.append(angle(predicted,se[i]))
            h=minrot(predicted,se[i])
            if h is None: failed=True; break
            if i == 2 and foot_mode in ('chain_transport_partial',
                                        'chain_transport_parent_coherent_partial'):
                amount=(rep.get('chain_diagnostics',{}).get(name,{})
                        .get('ankle_transport',{}))
                mu=amount.get('selected_mu')
                if not isinstance(mu,(int,float)) or not 0.0 < mu < 1.0:
                    failed=True; break
                h=scaled_minrot(h,float(mu))
            q=h@q
            if i==0: q_first=q.copy()
            expected_edges.append(q@de[i])
            expected=(q@trot[nodes[i]].to_matrix()).to_quaternion()
            rotation_errors.append(qerr(expected,orot[nodes[i]]))
        dirs=[angle(expected_edges[i],oe[i]) for i in range(len(expected_edges))]
        row={'mode':first_mode,'edge_direction_error_deg':dirs,
             'incremental_min_rotation_deg':increment_deg,
             'transport_rotation_error_deg':rotation_errors}
        if parent_coherent and q_first is not None:
            h_v31=minrot(de[0],se[0])
            if h_v31 is None:
                failed=True
                row['pelvis_boundary']={'pass':False,'reason':'independent V3.1 seed degenerate'}
            else:
                correction=q_first@h_v31.transposed()
                corr_deg=qerr(Matrix.Identity(3).to_quaternion(),correction.to_quaternion())
                corr_twist=signed_twist(correction,se[0])
                parent_dir=angle(q_first@de[0],se[0])
                reported=(rep.get('chain_diagnostics',{}).get(name,{})
                          .get('v31_boundary_correction_deg'))
                report_error=(None if reported is None else abs(reported-corr_deg))
                row['pelvis_boundary']={
                    'correction_deg':corr_deg,
                    'correction_twist_deg':corr_twist,
                    'non_twist_residual_deg':abs(corr_deg-abs(corr_twist)),
                    'parent_coherent_thigh_direction_error_deg':parent_dir,
                    'reported_correction_deg':reported,
                    'report_error_deg':report_error,
                    'pass':parent_dir is not None and parent_dir<=0.2
                           and report_error is not None and report_error<=0.2,
                }
                failed |= not row['pelvis_boundary']['pass']
        if foot_mode in ('chain_transport_partial',
                         'chain_transport_parent_coherent_partial') and len(oe) == 3:
            measured_residual=angle(se[2],oe[2])
            reported_residual=(rep.get('chain_diagnostics',{}).get(name,{})
                               .get('ankle_transport',{}).get('residual_direction_deg'))
            residual_error=(None if measured_residual is None or reported_residual is None
                            else abs(measured_residual-reported_residual))
            row['partial_foot']={'measured_residual_direction_deg':measured_residual,
                                 'reported_residual_direction_deg':reported_residual,
                                 'residual_error_deg':residual_error,
                                 'pass':residual_error is not None and residual_error<=0.2}
            failed |= not row['partial_foot']['pass']
        row['pass']=(not failed and all(x is not None and x<=0.2 for x in dirs)
                     and len(rotation_errors)==count and all(x<=0.2 for x in rotation_errors))
        out['chains'][name]=row; out['ok'] &= row['pass']
    # Active terminals must preserve the artist-authored target rest relationship.
    for side in ('L','R'):
        pairs=(('forearm.'+side,'hand.'+side),('leg.'+side,'foot.'+side),
               ('foot.'+side,'toe.'+side))
        for parent,child in pairs:
            mode=rep.get('solver_mode_by_bone',{}).get(child,'')
            if mode != 'terminal_follow':
                continue
            err=qerr(trot[parent].inverted()@trot[child],
                     orot[parent].inverted()@orot[child])
            key=parent+'->'+child
            out['terminal_relative'][key]={'mode':mode,'error_deg':err,'pass':err<=1.0}
            out['ok'] &= err<=1.0
    out['report_modes']=rep.get('solver_mode_by_bone',{})
    out['report_chain_fallbacks']=rep.get('chain_fallbacks',[])
    os.makedirs(os.path.dirname(os.path.abspath(a.json)) or '.',exist_ok=True)
    with open(a.json,'w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2); f.write('\n')
    print(('[OK]' if out['ok'] else '[FAIL]')+' verify_v32_transport_artifact '+a.json)
    print(json.dumps({'chains':out['chains'],'terminal_relative':out['terminal_relative']},ensure_ascii=False))
    return 0 if out['ok'] else 2


if __name__=='__main__': raise SystemExit(main())
