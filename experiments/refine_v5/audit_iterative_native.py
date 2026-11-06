"""Blender P4 actual target application, mesh audit, export/readback, exact fallback."""
import bpy,hashlib,json,sys
from pathlib import Path
from mathutils import Matrix,Quaternion,Vector
from mathutils.kdtree import KDTree

def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=str(path),ignore_leaf_bones=False)
    return next(o for o in bpy.context.scene.objects if o.type=='ARMATURE'),[o for o in bpy.context.scene.objects if o.type=='MESH']
def snapshot(arm):
    by={b.name.split(':')[-1]:b for b in arm.pose.bones}
    return dict(heads={n:list(arm.matrix_world@b.head) for n,b in by.items()},parents={n:b.parent.name.split(':')[-1] if b.parent else None for n,b in by.items()},lengths={n:(arm.matrix_world@(b.tail-b.head)).length for n,b in by.items()},matrices={n:[list(x) for x in arm.matrix_world@b.matrix] for n,b in by.items()})
def distance(a,b):
    tree=KDTree(len(b))
    for i,v in enumerate(b):tree.insert(Vector(v),i)
    tree.balance();return max(tree.find(Vector(v))[2] for v in a)
def main():
    run=Path(sys.argv[sys.argv.index('--')+1]);repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo))
    from converter import retarget as rt
    from converter.mesh_validation import extract_evaluated_skin,audit_mesh
    from src.refine_v4.skin_collision import inspect_self
    from experiments.refine_v5.export_native_v2 import export_preserving_bones
    from experiments.refine_v5.native_contract_v2 import compare_native
    from experiments.refine_v5.release_gate import finalize,sha
    from experiments.refine_v5.target_safety import skin_nonregression
    import numpy as np
    rigs={r['case_id']:r for r in json.loads((run/'native-data/records.json').read_text()) if r['status']=='ready'}
    rows=json.loads((run/'iterative-balanced/results.json').read_text());manifest=json.loads((run/'iterative-balanced/manifest.json').read_text());binding={r['case_id']:r for r in manifest['records']}
    cases=['001','006','074','029','100','022'];out=run/'native-audit';out.mkdir(exist_ok=False);records=[]
    ledger=[dict(case_id=r['case_id'],rounds=r['rounds'],status=r['status'],native_application_attempted=r['status']=='proposal') for r in rows if r['case_id'] in cases and r['stage']=='selected_target' and r.get('profile')=='experimental30']
    (out/'case-ledger.json').write_text(json.dumps(ledger,indent=2))
    for row in rows:
        if row['case_id'] not in cases or row['stage']!='selected_target' or row.get('profile')!='experimental30' or row['status']!='proposal':continue
        case=row['case_id'];rig=rigs[case];source=Path(rig['base_fbx']);before_sha=sha(source)
        arm,meshes=load(source);baseline=snapshot(arm);base_skin=inspect_self(arm)
        by={b.name.split(':')[-1]:b for b in arm.pose.bones}
        # Actual target geometry can reject or shrink a numerically good source proposal.
        trials=[]
        for fraction in [1.,.5,.25,0.]:
            for n,v in zip(row['active_bones'],row['delta_rotvec']):
                vec=Vector(v)*fraction;angle=vec.length;by[n].matrix_basis=Quaternion(vec/angle,angle).to_matrix().to_4x4() if angle>1e-12 else Matrix.Identity(4)
            bpy.context.view_layer.update();triage=inspect_self(arm)
            reasons=skin_nonregression(base_skin,triage);trials.append(dict(fraction=fraction,reasons=reasons))
            if not reasons:break
        planned=snapshot(arm)
        # Independent FK of the scaled ORIGINAL local changes, including all descendants.
        worlds={};rotations={n:by[n].matrix_basis.copy() for n in row['active_bones']}
        for b in rig['bones']:
            n=b['name'];parent=b['parent'];m=np.array(b['matrix'])
            if parent:m=np.linalg.solve(np.array(next(x['matrix'] for x in rig['bones'] if x['name']==parent)),m)
            if n in rotations:m=m@np.array(rotations[n])
            worlds[n]=worlds[parent]@m if parent else m
        planned_error=max(np.linalg.norm(np.array(planned['heads'][n])-worlds[n][:3,3]) for n in by)
        if planned_error>1e-5:raise ValueError('independent Blender/FK mismatch')
        vertices,faces=extract_evaluated_skin(meshes)
        strict=audit_mesh(vertices,faces,seconds=2.,profile_sha256=binding[case]['profile']['body_asset_sha256'])
        dest=out/(case+'.r'+str(row['rounds'])+'.diagnostic.fbx')
        rt.apply_output_mode(arm,meshes,'rigged_rest');export_preserving_bones(dest)
        arm,meshes=load(dest);actual=snapshot(arm)
        structural=compare_native(planned['heads'],actual['heads'],planned['parents'],actual['parents'],baseline['lengths'],actual['lengths'])
        # Bone orientation is necessary for palm/head/sole, even when heads are fixed.
        import numpy as np
        rotation_error=max(float(np.max(abs(np.array(planned['matrices'][n])[:3,:3]-np.array(actual['matrices'][n])[:3,:3]))) for n in by)
        av,af=extract_evaluated_skin(meshes);cloud=max(distance(vertices,av),distance(av,vertices))
        if rotation_error>2e-4 or cloud>1e-5:raise ValueError('native frame/skin roundtrip regression')
        if sha(source)!=before_sha:raise ValueError('source mutated')
        profile_digest=hashlib.sha256(json.dumps(binding[case]['profile'],sort_keys=True).encode()).hexdigest()
        audit=dict(source_sha256=before_sha,candidate_sha256=sha(dest),native_validated=True,skin=strict)
        release=finalize(source,dest,out/(case+'.r'+str(row['rounds'])+'.safe.fbx'),audit,profile_digest)
        records.append(dict(case_id=case,rounds=row['rounds'],artifact=str(dest),sha256=sha(dest),source_sha256=before_sha,preserved_bones=len(by),independent_fk_error_m=planned_error,native=structural,max_rotation_element_error=rotation_error,max_skin_cloud_error_m=cloud,planned_vertices=len(vertices),readback_vertices=len(av),baseline_skin_triage=base_skin,candidate_skin_triage=triage,full_skin_audit=strict,release=release,target_backtracking=trials,accepted_fraction=fraction,production_ready=False))
        (out/'records.json').write_text(json.dumps(records,indent=2));print('V5_NATIVE_AUDIT',case,row['rounds'],release['status'],flush=True)
if __name__=='__main__':main()
