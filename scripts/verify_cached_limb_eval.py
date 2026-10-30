"""Refresh four-limb measurements from the actual independently admitted files."""
import sys,json,hashlib
from pathlib import Path
repo,root,run=map(lambda p:Path(p).resolve(),sys.argv[sys.argv.index('--')+1:])
sys.path.insert(0,str(repo))
from converter import retarget as rt
from src.arm_refine.native_state import read_native_state
from src.arm_refine.diagnosis import diagnose
from src.experimental.native_leg_diagnosis import native_state as read_legs
from src.experimental.cached_leg_diagnosis import diagnose as diagnose_legs

def load(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

rows=[]
for case in load(run/'cases.json'):
    folder=run/case['id'];verification=load(folder/'export-verification.json')
    assert sha(root/case['source_fbx'])==case['source_sha256']==sha(folder/'before.fbx')
    assert sha(folder/'after.fbx')==verification['final_fbx_sha256']
    rt.reset_scene();arm,meshes=rt.import_character(str(folder/'after.fbx'))
    state=read_native_state(arm,meshes,load(folder/'target.json'))
    final=diagnose(load(folder/'target.json'),state)
    save(folder/'native-after.json',state);save(folder/'diagnosis-after.json',final)
    if (folder/'leg-target.json').is_file():
        legs=read_legs(arm,meshes);save(folder/'leg-native-after.json',legs)
        save(folder/'leg-diagnosis-after.json',diagnose_legs(load(folder/'leg-target.json'),legs))
    selected=load(folder/'plan.json')['selection']
    semantic=all(m['status']=='match' for m in final['members'] if m['limb'] in [s+'_arm' for s in selected])
    if verification['final_applied']:assert semantic
    else:assert sha(folder/'before.fbx')==sha(folder/'after.fbx')
    row={'id':case['id'],'final_applied':verification['final_applied'],
        'selected_supported_goals_match':semantic if verification['final_applied'] else None,
        'final_fbx_sha256':sha(folder/'after.fbx'),'read_only_matrix_delta':state['read_only_matrix_delta']}
    rows.append(row)
save(run/'final-limb-verification.json',rows)
save(run/'verification-implementation-hashes.json',{str(Path(__file__).relative_to(repo)):sha(Path(__file__))})
print(json.dumps({'people':len(rows),'applied':sum(r['final_applied'] for r in rows),'final_files_remeasured':True}))
