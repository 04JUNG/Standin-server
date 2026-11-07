"""Controlled, offline CS3C skin-deformation diagnosis; no production edits.

Same-pose DQ/weight smoothing are experiments, not approved replacement assets.
Rotation ablations intentionally change the pose and only localize sensitivity.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_actual_mesh import read, topology, collisions, areas, quantiles


def evaluated(obj):
    dep=bpy.context.evaluated_depsgraph_get()
    value=obj.evaluated_get(dep);mesh=value.to_mesh()
    try:return np.array([value.matrix_world@v.co for v in mesh.vertices],dtype=float)
    finally:value.to_mesh_clear()


def dense_weights(obj):
    names=[g.name for g in obj.vertex_groups]
    w=np.zeros((len(obj.data.vertices),len(names)))
    for vertex in obj.data.vertices:
        for g in vertex.groups:w[vertex.index,g.group]=g.weight
    return names,w


def set_weights(obj,names,w):
    ids=list(range(len(obj.data.vertices)))
    for group in obj.vertex_groups:group.remove(ids)
    for j,name in enumerate(names):
        group=obj.vertex_groups[name]
        for i in np.flatnonzero(w[:,j]>1e-9):group.add([int(i)],float(w[i,j]),'REPLACE')
    bpy.context.view_layer.update()


def smooth_weights(weights,edges,iterations):
    degree=np.bincount(edges.flatten(),minlength=len(weights))
    w=weights.copy()
    for _ in range(iterations):
        accum=np.zeros_like(w)
        np.add.at(accum,edges[:,0],w[edges[:,1]])
        np.add.at(accum,edges[:,1],w[edges[:,0]])
        w=.5*w+.5*accum/np.maximum(degree[:,None],1)
        w/=np.maximum(w.sum(1,keepdims=True),1e-12)
    return w


def metrics(vertices,faces,rest,edges,rest_hits,labels):
    lengths=np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1)
    posed=np.linalg.norm(vertices[edges[:,1]]-vertices[edges[:,0]],axis=1)
    ratios=posed/lengths
    hits,_=collisions(vertices,faces)
    ar=areas(vertices,faces)/areas(rest,faces)
    region=Counter()
    adjacent=0
    for a,b in hits-rest_hits:
        adjacent+=bool(set(faces[a])&set(faces[b]))
        bones={Counter(labels[int(i)] for i in faces[k]).most_common(1)[0][0] for k in (a,b)}
        for bone in bones:region[bone]+=1
    return {'maxEdgeRatio':float(ratios.max()),'p99EdgeRatio':float(np.percentile(ratios,99)),
            'edgesOverTwoTimes':int((ratios>2).sum()),'edgesBelowOneFifth':int((ratios<.2).sum()),
            'newStrictCrossingPairs':len(hits-rest_hits),'totalStrictCrossingPairs':len(hits),
            'facesBelowOnePercentArea':int((ar<.01).sum()),'minFaceAreaRatio':float(ar.min()),
            'newCrossingPairRegions':dict(region),'newCrossingPairsSharingVertex':adjacent}


def angle(a,b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))


def bend(arm,tbl,roles):
    points=[np.array(arm.matrix_world@arm.pose.bones[tbl[r]].head) for r in roles]
    return angle(points[1]-points[0],points[2]-points[1])


def edge_explanation(rest,posed,edges,weights,names,base_frames,pose_frames):
    ratios=np.linalg.norm(posed[edges[:,1]]-posed[edges[:,0]],axis=1)/np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1)
    normalized=weights/weights.sum(1,keepdims=True)
    transform={name:pose_frames[name]@np.linalg.inv(base_frames[name]) for name in names if name in base_frames}
    result=[]
    for k in np.argsort(ratios)[-10:][::-1]:
        i,j=map(int,edges[k]);mid=(rest[i]+rest[j])/2;delta=rest[j]-rest[i]
        smooth=np.zeros(3);gradient=np.zeros(3);terms=[]
        for b,name in enumerate(names):
            wi,wj=normalized[i,b],normalized[j,b]
            if wi==wj==0:continue
            t=transform[name]
            smooth+=(wi+wj)/2*(t[:3,:3]@delta)
            contribution=(wj-wi)*(t[:3,:3]@mid+t[:3,3])
            gradient+=contribution
            terms.append({'bone':name,'weightStart':float(wi),'weightEnd':float(wj),
                          'gradientTermVectorM':contribution.tolist()})
        # Exact identity: weighted rigid edge term + weight-gradient term.
        residual=float(np.linalg.norm(smooth+gradient-(posed[j]-posed[i])))
        assert residual<1e-5,'Independent edge decomposition did not reproduce the actual edge'
        assert np.linalg.norm(smooth)<=np.linalg.norm(delta)*(1+1e-5),'Rigid weighted edge term is not contractive'
        result.append({'vertices':[i,j],'restEdgeM':float(np.linalg.norm(delta)),
                       'restEndpointsM':[rest[i].tolist(),rest[j].tolist()],
                       'posedEdgeM':float(np.linalg.norm(posed[j]-posed[i])), 'ratio':float(ratios[k]),
                       'weightL1Difference':float(abs(normalized[j]-normalized[i]).sum()),
                       'rigidEdgeTermM':float(np.linalg.norm(smooth)),
                       'weightGradientTermM':float(np.linalg.norm(gradient)),
                       'edgeDecompositionResidualM':residual, 'weights':terms})
    return result


def source_comparison(path,rest,reference_weights):
    if path.suffix=='.blend':
        bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False,use_scripts=False)
        obj=next(o for o in bpy.context.scene.objects if o.type=='MESH')
        vertices=np.array([obj.matrix_world@v.co for v in obj.data.vertices])
        names,w=dense_weights(obj)
        weights=[{names[j]:float(w[i,j]) for j in np.flatnonzero(w[i]>0)} for i in range(len(w))]
    else:
        data=read(path);vertices=data[0];weights=[dict(w) for w in data[3]]
    comparable=len(vertices)==len(rest)
    surface_error=float(np.linalg.norm(vertices-rest,axis=1).max()) if comparable else None
    # Direct index correspondence is admissible only for identical rest shapes.
    comparable=comparable and surface_error<2e-5
    error=max(abs(dict(reference_weights[i]).get(name,0)-weights[i].get(name,0))
              for i in range(len(weights)) for name in set(dict(reference_weights[i]))|set(weights[i])) if comparable else None
    return {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'vertexCount':len(vertices),'maxRestVertexErrorM':surface_error,
            'directVertexCorrespondence':bool(comparable),'maxWeightDifference':error}


def diagnose(directory,runtime,poses,source_paths):
    sys.path.insert(0,str(runtime))
    from converter import retarget as rt
    from converter import bone_map
    from converter.local import inspection
    base=read(directory/'model-0-registered.fbx')
    rest,faces,_,_,labels,base_frames=base
    comparisons=[source_comparison(path,rest,base[3]) for path in source_paths]
    _,edges=topology(rest,faces);rest_hits,_=collisions(rest,faces)
    outputs=[]
    for p in range(2):
        posed=read(directory/f'model-0-pose-{p}.fbx')
        out_vertices=posed[0];out_frames=posed[5]
        bpy.ops.wm.read_factory_settings(use_empty=True)
        arm,meshes=inspection.import_fbx(directory/'model-0-registered.fbx')
        inspection.inspect(arm,meshes)
        mesh=meshes[0]
        names,weights=dense_weights(mesh)
        rt._apply_object_transform(arm)
        src=rt.import_bvh(str(poses[p]),frame=0)
        report=rt.retarget(src,arm,dst_profile='mixamo',apply_root_translation=False)
        assert report.ok
        bpy.context.view_layer.update()
        before_bake=evaluated(mesh)
        reproduction=float(np.linalg.norm(before_bake-out_vertices,axis=1).max())
        print('REPRODUCTION_MAX_VERTEX_ERROR_M',reproduction,flush=True)
        assert reproduction<2e-5,'Failed to reproduce actual output before FBX baking'
        saved={b.name:b.matrix_basis.copy() for b in arm.pose.bones}
        saved_world={b.name:(arm.matrix_world@b.matrix).copy() for b in arm.pose.bones}
        joints={b.name:(arm.matrix_world@b.head).copy() for b in arm.pose.bones}
        tbl=bone_map.PROFILES['mixamo'];stbl=bone_map.PROFILES[report.src_profile]
        chains={'elbow.L':('upperarm.L','forearm.L','hand.L'),'elbow.R':('upperarm.R','forearm.R','hand.R'),
                'knee.L':('upleg.L','leg.L','foot.L'),'knee.R':('upleg.R','leg.R','foot.R')}
        bends={name:{'source':bend(src,stbl,roles),'target':bend(arm,tbl,roles)} for name,roles in chains.items()}
        channels={}
        source_channels={}
        for role,name in tbl.items():
            q=saved[name].to_quaternion();swing,twist=q.to_swing_twist('Y')
            channels[role]={'rotationDeg':float(np.degrees(q.angle)),
                            'swingDeg':float(np.degrees(swing.angle)),'twistDeg':float(np.degrees(twist))}
            sq=src.pose.bones[stbl[role]].matrix_basis.to_quaternion()
            ss,st=sq.to_swing_twist('Y')
            source_channels[role]={'rotationDeg':float(np.degrees(sq.angle)),
                                   'swingDeg':float(np.degrees(ss.angle)),'twistDeg':float(np.degrees(st))}
        variants=[]
        def measure(label):
            bpy.context.view_layer.update();v=evaluated(mesh)
            delta=max(((arm.matrix_world@b.head)-joints[b.name]).length for b in arm.pose.bones)
            row={'experiment':label,'maxJointHeadChangeM':delta,**metrics(v,faces,rest,edges,rest_hits,labels)}
            variants.append(row);print('DEFORMATION_EXPERIMENT',p,label,json.dumps(row),flush=True)
            return v
        measure('original_lbs_before_bake')
        mod=next(m for m in mesh.modifiers if m.type=='ARMATURE')
        assert not mod.use_deform_preserve_volume
        mod.use_deform_preserve_volume=True
        dq=measure('same_pose_original_weights_dual_quaternion')
        assert variants[-1]['maxJointHeadChangeM']<2e-6
        mod.use_deform_preserve_volume=False
        for iterations in (1,3):
            smoothed=smooth_weights(weights,edges,iterations)
            set_weights(mesh,names,smoothed)
            measure(f'same_pose_lbs_neighbor_weight_smoothing_{iterations}')
            assert variants[-1]['maxJointHeadChangeM']<2e-6
        set_weights(mesh,names,weights)
        # Change only a named local joint's rotation, retaining translation and
        # scale. These are causal perturbations, not pose-preserving fixes.
        for label,roles in [('neutral_thighs',['upleg.L','upleg.R']),
                            ('neutral_knees',['leg.L','leg.R']),
                            ('neutral_elbows',['forearm.L','forearm.R']),
                            ('neutral_upperarms',['upperarm.L','upperarm.R']),
                            ('remove_local_thigh_twist',['upleg.L','upleg.R']),
                            ('remove_local_elbow_twist',['forearm.L','forearm.R']),
                            ('remove_local_upperarm_twist',['upperarm.L','upperarm.R'])]:
            for b in arm.pose.bones:b.matrix_basis=saved[b.name]
            for role in roles:
                name=tbl[role];loc,q,scale=saved[name].decompose()
                rotation=q.to_swing_twist('Y')[0] if label.startswith('remove') else Quaternion()
                arm.pose.bones[name].matrix_basis=Matrix.LocRotScale(loc,rotation,scale)
            measure(label)
        # Hold all joint heads and downstream world frames fixed. Change only
        # upperarm roll: seed minimum swing from its already solved shoulder,
        # instead of the world identity used by active arm chain transport.
        for b in arm.pose.bones:b.matrix_basis=saved[b.name]
        bpy.context.view_layer.update()
        roll_details={}
        for side in ('L','R'):
            role='upperarm.'+side
            if report.solver_mode_by_bone[role]!='chain_transport':continue
            name=tbl[role];bone=arm.pose.bones[name];parent=bone.parent
            rest_rotation=rt._rot_only(rt._rest_world(arm,name))
            seed=rt._rot_only(saved_world[parent.name])@rt._rot_only(rt._rest_world(arm,parent.name)).inverted()
            before=rt._rot_only(saved_world[name])
            predicted=(seed@rest_rotation).to_3x3().col[1]
            wanted=before.to_3x3().col[1]
            correction,reason=rt._min_rotation(predicted,wanted)
            assert correction is not None,reason
            desired=correction@seed@rest_rotation
            desired.translation=saved_world[name].translation
            bone.matrix=arm.matrix_world.inverted()@desired
            bpy.context.view_layer.update()
            for child in bone.children_recursive:
                child.matrix=arm.matrix_world.inverted()@saved_world[child.name]
                bpy.context.view_layer.update()
            q=bone.matrix_basis.to_quaternion()
            roll_details[role]={'beforeLocalTwistDeg':channels[role]['twistDeg'],
                                'afterLocalTwistDeg':float(np.degrees(q.to_swing_twist('Y')[1]))}
        measure('parent_seeded_upperarm_roll_downstream_world_held' if roll_details else 'no_op_basis_restore_control')
        # An exact endpoint-weight decomposition explains why a connected edge
        # elongates even though every participating bone transform is rigid.
        explanation=edge_explanation(rest,out_vertices,edges,weights,names,base_frames,out_frames)
        sums=weights.sum(1)
        l1=abs(weights[edges[:,0]]/sums[edges[:,0],None]-weights[edges[:,1]]/sums[edges[:,1],None]).sum(1)
        row={'pose':poses[p].name,'sourceProfile':report.src_profile,
             'preBakeVsActualFbxMaxVertexErrorM':reproduction,
             'maxJointEdgeLengthErrorM':json.loads((directory/'report.json').read_text())['models'][0]['poses'][p]['retarget']['max_joint_edge_length_error'],
             'jointBendsDeg':bends,'targetLocalRotations':channels,
             'sourceLocalRotations':source_channels,
             'weightSum':quantiles(sums),'edgeWeightL1Difference':quantiles(l1),
             'verticesWithMoreThanFourWeights':int((np.count_nonzero(weights>1e-8,axis=1)>4).sum()),
             'worstEdges':explanation,'experiments':variants,
             'solverModes':report.solver_mode_by_bone,
             'restSwingDeg':report.rest_swing_deg}
        row['parentSeededRollDetails']=roll_details
        outputs.append(row)
    return {'version':'actual-cs3c-deformation-diagnosis/1',
            'scope':'two unique geometries from the previous eight-file audit; same male0.7.6',
            'experimentsAreProductionFixes':False,'sourceComparisons':comparisons,
            'numericalCaveat':'Reassigning float32 bone bases can alter a few near-boundary crossings. Compare substantive changes against the pre-bake baseline; no-op control is included.',
            'poses':outputs}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--pose',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-control',type=Path,action='append',default=[])
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);assert len(args.pose)==2
    result=diagnose(args.run_dir,args.runtime,args.pose,args.source_control)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
