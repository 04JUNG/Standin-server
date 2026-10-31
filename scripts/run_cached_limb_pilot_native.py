"""Blender offline auto diagnosis -> automatic arm plan -> native correction.

No network libraries or VLM SDK. Uses only prepared cached target evidence.
"""
import sys,json,time,hashlib,shutil,argparse
from pathlib import Path
parser=argparse.ArgumentParser(description='Disabled by default. Offline experimental cached-limb diagnosis only.')
parser.add_argument('repo',type=Path);parser.add_argument('workspace',type=Path);parser.add_argument('run',type=Path)
parser.add_argument('--enable-experimental-diagnosis',action='store_true')
cli_args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
args=parser.parse_args(cli_args)
if not args.enable_experimental_diagnosis:parser.error('experimental diagnosis is disabled; explicit --enable-experimental-diagnosis required')
repo,root,run=(p.resolve() for p in (args.repo,args.workspace,args.run))
import bpy
import numpy as np
sys.path.insert(0,str(repo));sys.path.insert(0,str(root/'pose-complement-1000-20261015'))
from converter import retarget as rt
from self_collision import inspect_self
from src.arm_refine.diagnosis import diagnose
from src.arm_refine.native_state import read_native_state
from src.arm_refine.language_plan import plan_language_arms
from src.arm_refine.language_adapter import apply_language_plan
from src.arm_refine.contact_adapter import bones
from src.arm_refine.native_export import bake_arm_rest
from src.arm_refine.canonical_surface import capture_canonical,rebuild_selected_surface
from src.experimental.cached_leg_diagnosis import diagnose as diagnose_legs
from src.experimental.native_leg_diagnosis import native_state as read_legs

def load(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
started=time.perf_counter()
rt.reset_scene();master,meshes=rt.import_character(str(root/'pose-expansion-20261015/mesh-review/assets/standin-master-v2.fbx'))
canonical=capture_canonical(master,meshes)
save(run/'canonical-source.json',{'path':'pose-expansion-20261015/mesh-review/assets/standin-master-v2.fbx',
    'sha256':sha(root/'pose-expansion-20261015/mesh-review/assets/standin-master-v2.fbx')})
b=bones(master);profile={'wrist_rest_relative':{s:[list(v) for v in b[s+'ForeArm'].matrix.to_3x3().normalized().transposed()@b[s+'Hand'].matrix.to_3x3().normalized()] for s in ('Left','Right')}}
save(run/'rig-profile.json',profile);requests=[];summary=[]
for case in load(run/'cases.json'):
    start=time.perf_counter();folder=run/case['id'];source=root/case['source_fbx'];assert sha(source)==case['source_sha256']
    target=load(folder/'target.json');shutil.copyfile(source,folder/'before.fbx')
    before_import=time.perf_counter();rt.reset_scene();arm,meshes=rt.import_character(str(source));import_seconds=time.perf_counter()-before_import
    base_times=[]
    for _ in range(15):
        tick=time.perf_counter();read_native_state(arm,meshes);base_times.append(time.perf_counter()-tick)
    tick=time.perf_counter();state=read_native_state(arm,meshes,target);state_seconds=time.perf_counter()-tick
    save(folder/'native-before.json',state)
    tick=time.perf_counter();diagnosis=diagnose(target,state);comparison_seconds=time.perf_counter()-tick
    save(folder/'diagnosis-before.json',diagnosis)
    if (folder/'leg-target.json').is_file():
        tick=time.perf_counter();leg_state=read_legs(arm,meshes);leg_seconds=time.perf_counter()-tick
        save(folder/'leg-native-before.json',leg_state)
        save(folder/'leg-diagnosis-before.json',diagnose_legs(load(folder/'leg-target.json'),leg_state))
    plan=None;receipt={'accepted':False,'status':'held','reasons':['no_safe_automatic_arm_correction'],'members':[]}
    correction_seconds=0.
    if diagnosis['selection']:
        plan=plan_language_arms(diagnosis['clauses'],diagnosis['selection'],source='cached_vlm_reference_with_local_geometry_diagnosis',
            selection_source=diagnosis['selection_source'])
        tick=time.perf_counter()
        try:receipt=apply_language_plan(arm,meshes,plan,inspect_self,profile)
        except (ValueError,KeyError,RuntimeError) as error:
            receipt={'accepted':False,'status':'held','members':[],'reasons':['native_fit_exception_'+type(error).__name__]}
        correction_seconds=time.perf_counter()-tick
    else:
        plan={'ready':False,'members':[],'selection':[],'automatic_change_selection':True,'reasons':receipt['reasons']}
    save(folder/'plan.json',plan);save(folder/'contact-validation.json',receipt)
    rebuild_sides=[m['side'] for m in plan['members'] if m.get('active') and m['action']=='forehead_straight']
    if receipt['accepted']:
        bake_arm_rest(arm,meshes,sides=diagnosis['selection'])
        if rebuild_sides:
            try:save(folder/'surface-rebuild.json',rebuild_selected_surface(arm,meshes,canonical,rebuild_sides))
            except ValueError:
                receipt.update(accepted=False,status='held',reasons=['canonical_surface_not_compatible'])
    if receipt['accepted']:
        rt.export_fbx(str(folder/'after.fbx'))
        bpy.ops.wm.save_as_mainfile(filepath=str(folder/'after.blend'))
    else:
        shutil.copyfile(source,folder/'after.fbx')
        rt.reset_scene();arm,meshes=rt.import_character(str(source));bpy.ops.wm.save_as_mainfile(filepath=str(folder/'after.blend'))
    rt.reset_scene();arm,meshes=rt.import_character(str(folder/'after.fbx'))
    state_after=read_native_state(arm,meshes,target);save(folder/'native-after.json',state_after)
    after_diagnosis=diagnose(target,state_after)
    if receipt['accepted'] and any(m['status']!='match' for m in after_diagnosis['members'] if m['limb'] in [s+'_arm' for s in diagnosis['selection']]):
        shutil.copyfile(folder/'after.fbx',folder/'semantic-rejected.fbx')
        shutil.copyfile(source,folder/'after.fbx');rt.reset_scene();arm,meshes=rt.import_character(str(source))
        bpy.ops.wm.save_as_mainfile(filepath=str(folder/'after.blend'))
        state_after=read_native_state(arm,meshes,target);save(folder/'native-after.json',state_after)
        after_diagnosis=diagnose(target,state_after)
        receipt.update(accepted=False,status='held',reasons=['saved_selected_limb_does_not_satisfy_diagnosed_goal'])
    save(folder/'contact-validation.json',receipt)
    save(folder/'diagnosis-after.json',after_diagnosis)
    request={**case,'selection':diagnosis['selection'],'clauses':diagnosis['clauses'],
             'canonical_surface_rebuild':bool(rebuild_sides),
             'description_source':'cached_vlm_reference_with_local_geometry_diagnosis',
             'selection_source':diagnosis['selection_source'],'previous_fbx':str((folder/'before.fbx').relative_to(root)),
             'comparison_baseline_fbx':str((folder/'before.fbx').relative_to(root))}
    requests.append(request);save(run/'requests.json',requests)
    row={'id':case['id'],'automatically_selected_sides':diagnosis['selection'],'accepted':receipt['accepted'],
         'status':receipt['status'],'reasons':receipt['reasons'],'final_fbx_sha256':sha(folder/'after.fbx'),
         'byte_exact_fallback':sha(source)==sha(folder/'after.fbx') if not receipt['accepted'] else None,
         'timing':{'fbx_import_seconds':import_seconds,'base_geometry_p50_ms':float(np.median(base_times)*1000),
                   'target_contact_read_seconds':state_seconds,'comparison_ms':comparison_seconds*1000,
                   'correction_seconds':correction_seconds,'case_total_seconds':time.perf_counter()-start},
         'decisions_before':{r['limb']:r['status'] for r in diagnosis['members']},'new_api_calls':0}
    summary.append(row);save(run/'summary.json',summary);print(json.dumps(row,ensure_ascii=False),flush=True)
save(run/'native-timing.json',{'batch_wall_seconds':time.perf_counter()-started,'cases':len(summary),'new_api_calls':0,
    'new_api_tokens':0,'image_recognition_time_included':False,'preview_rendering_time_included':False})
