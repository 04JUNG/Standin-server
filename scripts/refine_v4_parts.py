"""Opt-in native part runner. No production API or search defaults are changed.

Blender --background --python-exit-code 1 --python scripts/refine_v4_parts.py --
    requests.json NEW_OUTPUT
Paths in requests are relative to the request file. Current asset fingerprints
and side attribution are mandatory; rejected edits return that exact asset.
"""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from converter import retarget as rt
from src.refine_v4.current_asset import current_source
from src.refine_v4.pipeline import Stage,run_refine_v4,snapshot,skin_nonregression
from src.refine_v4.native_geometry import extract_geometry_part,apply_geometry_part,advance_foot
from src.refine_v4.native_export import bake_selected_rest
from src.refine_v4.leg_adapter import plan_legs,apply_native_leg_part
from src.refine_v4.skin_collision import inspect_self


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def surface(meshes):
    graph=bpy.context.evaluated_depsgraph_get();points=[]
    for m in meshes:
        obj=m.evaluated_get(graph);data=obj.to_mesh()
        try:points.extend(list(obj.matrix_world@v.co) for v in data.vertices)
        finally:obj.to_mesh_clear()
    return np.asarray(points)


def execute(req,root,folder):
    source=current_source(req,root);folder.mkdir();shutil.copy2(source,folder/'before.fbx')
    result={'id':req['id'],'accepted':False,'status':'not_applied','source_sha256':sha(source),
            'score_status':'not_scored','release_allowed':False,'stages':[],'reasons':[]}
    rt.reset_scene();arm,meshes=rt.import_character(str(source))
    original=snapshot(arm);baseline=inspect_self(arm);result['initial_skin']=baseline
    try:
        method=req.get('method','retain_current')
        if method=='retain_current':
            if req.get('arms'):raise ValueError('retain_current cannot request an arm edit')
            # Visual approval preserves an authored file; it does not certify
            # pre-existing skin flags or claim a new automatic success.
            result.update(accepted=True,status='retained',automatic_saved_edit=False)
            shutil.copy2(source,folder/'after.fbx');return result
        if not skin_nonregression(baseline,baseline)['accepted']:
            raise ValueError('complete baseline skin evidence required')
        if method=='native_geometry':
            donor=root/req['part']['fbx']
            if sha(donor)!=req['part']['sha256']:raise ValueError('donor fingerprint mismatch')
            # Extract in an isolated scene: skin checks must not include donor.
            rt.reset_scene();a,m=rt.import_character(str(donor))
            part=extract_geometry_part(a,m,sides=req['part']['sides'],source_sha256=sha(donor),intent=req['part']['intent'])
            rt.reset_scene();arm,meshes=rt.import_character(str(source))
            receipt=apply_geometry_part(arm,meshes,part,skin_check=inspect_self)
            result['stages'].append({'kind':'native_geometry','receipt':receipt})
            if not receipt['accepted']:raise ValueError('native geometry part rejected')
        leg=None
        if method=='advance_foot':
            side=req['foot']['side'];distance=req['foot']['distance_m']
            leg=Stage('legs',(side,),lambda a,m:advance_foot(a,side=side,distance_m=distance))
        elif method=='directions':
            spec=req['legs'];plan=plan_legs(spec['sentence'],selection=spec['selection'],source=spec['source'],reference=spec['reference'])
            part=json.loads((root/spec['part_file']).read_text())
            leg=Stage('legs',tuple(plan['sides']),lambda a,m:apply_native_leg_part(a,m,part,plan,skin_check=inspect_self))
        elif method not in ('native_geometry','arms'):
            raise ValueError('unsupported explicit part method')
        arms=None;profile=None
        if req.get('arms'):
            from src.arm_refine.v4_adapter import apply_arm_stage,check_arm_stage
            profile=json.loads((root/req['rig_profile']).read_text())
            arms=Stage('arms',('left','right'),lambda a,m:apply_arm_stage(a,m,clauses=req['arms'],skin_check=inspect_self,rig_profile=profile))
        if method=='arms' and arms is None:raise ValueError('arm clauses required')
        def final(a,m,receipts):
            skin=skin_nonregression(baseline,inspect_self(a));checks=[]
            for item in receipts:
                if item['kind']=='arms':checks.append(check_arm_stage(a,m,item['receipt'],skin_check=inspect_self,rig_profile=profile))
            return {'accepted':skin['accepted'] and all(c['accepted'] for c in checks),
                    'reasons':skin['reasons']+[reason for c in checks for reason in c['reasons']],
                    'skin':skin,'arms':checks}
        receipts=[]
        if leg or arms:
            transaction=run_refine_v4(arm,meshes,leg_stage=leg,arm_stage=arms,final_check=final)
            save(folder/'transaction.json',transaction);result['stages']+=transaction['stages'];receipts=transaction['stages']
            if not transaction['accepted']:raise ValueError('part transaction rejected: '+str(transaction['reasons']))
        expected=snapshot(arm);expected_surface=surface(meshes)
        selected=[p.name for p in arm.pose.bones if np.max(abs(np.array(p.matrix_basis)-np.eye(4)))>1e-6]
        if selected:bake_selected_rest(arm,meshes,selected)
        rt.export_fbx(str(folder/'candidate.fbx'))
        rt.reset_scene();arm,meshes=rt.import_character(str(folder/'candidate.fbx'))
        saved=snapshot(arm);joint_error=max(float(np.max(abs(saved[n]-v))) for n,v in expected.items())
        saved_surface=surface(meshes)
        if saved_surface.shape!=expected_surface.shape:raise ValueError('saved surface topology changed')
        vertex_error=float(np.max(np.linalg.norm(saved_surface-expected_surface,axis=1)))
        final_check=final(arm,meshes,receipts)
        result['saved_verification']={'matrix_error':joint_error,'surface_error_m':vertex_error,'final_check':final_check}
        if joint_error>2e-5 or vertex_error>2e-5 or not final_check['accepted']:
            raise ValueError('saved native artifact verification rejected')
        changed=max(float(np.max(abs(saved[n]-v))) for n,v in original.items())>1e-6 or method=='native_geometry'
        result.update(accepted=True,status='applied' if changed else 'unchanged',automatic_saved_edit=changed)
        shutil.copy2(folder/'candidate.fbx',folder/'after.fbx')
    except Exception as exc:
        result['reasons'].append(str(exc))
        shutil.copy2(source,folder/'after.fbx')
    return result


if __name__=='__main__':
    request_file,out=map(lambda v:Path(v).resolve(),sys.argv[sys.argv.index('--')+1:])
    requests=json.loads(request_file.read_text());ids=[r['id'] for r in requests]
    if len(set(ids))!=len(ids) or any(not s.replace('-','').isalnum() for s in ids):raise ValueError('unique simple request IDs required')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(request_file,out/'requests.json');rows=[]
    for req in requests:
        row=execute(req,request_file.parent,out/req['id']);row['after_sha256']=sha(out/req['id']/'after.fbx')
        save(out/req['id']/'receipt.json',row);rows.append(row);save(out/'summary.json',rows)
        print(req['id'],row['status'],row['reasons'],flush=True)
