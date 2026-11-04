"""Blender-only integration checks, using caller-supplied canonical FBX files.

-- source.fbx donor.fbx new-result.json
Artificial collision replies test rollback; they are not geometric evidence.
"""
import copy,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from converter import retarget as rt
from src.refine_v4.native_geometry import extract_geometry_part,apply_geometry_part,_digest
from src.refine_v4.pipeline import snapshot
from src.refine_v4.skin_collision import inspect_self
import hashlib
source,donor,out=map(Path,sys.argv[sys.argv.index('--')+1:])
rt.reset_scene();arm,meshes=rt.import_character(str(donor))
part=extract_geometry_part(arm,meshes,sides=['left','right'],source_sha256=hashlib.sha256(donor.read_bytes()).hexdigest(),intent={'source':'reviewed_library','reference':'canonical integration fixture'})
rows=[]
def start():
    rt.reset_scene();arm,meshes=rt.import_character(str(source));return arm,meshes,snapshot(arm),[np.array([v.co[:] for v in m.data.vertices]) for m in meshes]
def same(a,m,j,vertices):
    return max(float(np.max(abs(snapshot(a)[n]-v))) for n,v in j.items())<2e-5 and all(np.max(abs(np.array([v.co[:] for v in mesh.data.vertices])-co))<1e-8 for mesh,co in zip(m,vertices))
for mode in ['fingerprint','topology','weights','wrong_relation','missing_skin','new_skin','real_skin']:
    arm,meshes,j,vertices=start();p=copy.deepcopy(part)
    if mode=='fingerprint':p['source_sha256']='0'*64
    if mode=='topology':p['meshes'][0]['topology'][0].reverse();p['id']=_digest({k:v for k,v in p.items() if k!='id'})
    if mode=='weights':
        w=p['meshes'][0]['weights'][0];w[next(iter(w))]+=.1;p['id']=_digest({k:v for k,v in p.items() if k!='id'})
    if mode=='wrong_relation':
        p['intent'].update(top_side='right',cross_region='knee');p['id']=_digest({k:v for k,v in p.items() if k!='id'})
    calls=[0]
    def fake(_):
        calls[0]+=1
        if mode=='missing_skin':return {}
        return {'regions':14,'collisions':([] if calls[0]==1 else [{'regions':['LeftThigh','RightThigh'],'max_depth_mm':500}]),'region_closure_failures':{}}
    try:
        result=apply_geometry_part(arm,meshes,p,skin_check=fake if mode in ['missing_skin','new_skin'] else inspect_self)
    except ValueError as error:
        assert mode in ['fingerprint','topology','weights'] and same(arm,meshes,j,vertices)
        result={'accepted':False,'reasons':[str(error)],'unmodified':True}
    else:
        assert result['accepted']==(mode=='real_skin'),result
        if mode!='real_skin':assert same(arm,meshes,j,vertices)
    rows.append({'mode':mode,**result});print('PASS',mode,flush=True)
out.write_text(json.dumps({'passed':True,'cases':rows,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'donor_sha256':hashlib.sha256(donor.read_bytes()).hexdigest()},ensure_ascii=False,indent=2)+'\n')
