"""Reject increased tested-region crossings relative to the frozen 004+010 baseline."""
import sys,json,shutil,runpy
from pathlib import Path
import numpy as np
from mathutils.bvhtree import BVHTree
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from validity import shadow, check_fit

root,baseline,assets=map(Path,sys.argv[sys.argv.index('--')+1:])
def crossings(vertices, faces, weights, names):
    groups={}
    for side in ('Left','Right'):
        for region,bones in [('upperarm',[side+'Arm']),('forearm',[side+'ForeArm']),('hand',[n for n in names if n.startswith(side+'Hand')]),('thigh',[side+'UpLeg']),('shin',[side+'Leg']),('foot',[n for n in names if n.startswith(side+'Foot') or n.startswith(side+'Toe')])]:
            groups[side+region]=bones
    groups['torso']=['Hips','Spine','Spine1','Spine2','Neck']
    patches={}
    for name,bones in groups.items():
        mass=weights[:,[names.index(n) for n in bones]].sum(axis=1)
        # Conservative weighted interiors avoid counting neighboring joint seams.
        patch=faces[np.min(mass[faces],axis=1)>.6]
        if len(patch):patches[name]=BVHTree.FromPolygons(vertices.tolist(),patch.tolist(),all_triangles=True)
    pairs=[(s+r,'torso') for s in ('Left','Right') for r in ('upperarm','forearm','hand','shin')]
    pairs += [('Leftupperarm','Rightupperarm'),('Leftforearm','Rightforearm'),('Leftthigh','Rightthigh'),('Leftshin','Rightshin')]
    pairs += [(s+'foot',o+r) for s,o in [('Left','Right'),('Right','Left')] for r in ['shin','thigh','foot']]
    pairs += [('Leftshin','Rightthigh'),('Leftthigh','Rightshin')]
    return {f'{a}/{b}':len(patches[a].overlap(patches[b])) for a,b in pairs if a in patches and b in patches}

rows=json.loads((root/'results.json').read_text());audit=[]
for row in rows:
 cid=row['case_id']
 if row['status']!='fit':continue
 dest=root/cid/'fit';old=baseline/cid/'fit'
 rig=json.loads((assets/'input/p1-native'/f'{cid}-source.json').read_text());names=[b['name'] for b in rig['bones']];ix={n:i for i,n in enumerate(names)};base=np.array([b['matrix'] for b in rig['bones']])
 with np.load(assets/'base-mesh'/f'{cid}.npz') as m:v,f,weights=m['vertices'],m['faces'],m['weights']
 check_fit(dest,rig,assets/'input'/rig['source'])
 with np.load(old/'expected-world.npz') as d:b=d['world']
 with np.load(dest/'expected-world.npz') as d:w=d['world']
 def skin(world):return np.einsum('vn,nvi->vi',weights,np.einsum('nij,vj->nvi',world@np.linalg.inv(base),np.c_[v,np.ones(len(v))])[:,:,:3])
 before=crossings(skin(b),f,weights,names);after=crossings(skin(w),f,weights,names);increases={k:after[k]-before[k] for k in before if after[k]>before[k]}
 met=json.loads((dest/'metrics.json').read_text());met['collision_screen']=dict(before=before,proposal=after,increases=increases,certified=False)
 if increases and shadow():met['collision_screen']['would_rollback']=True
 if increases and not shadow():
  rejected=root/cid/'collision-rejected';rejected.mkdir(exist_ok=True)
  for name in ['adjustment.json','expected-world.npz','camera.json']:shutil.copy2(dest/name,rejected/name)
  for name in ['adjustment.json','expected-world.npz']:shutil.copy2(old/name,dest/name)
  pr=json.loads((old/'adjustment.json').read_text());anchor=((b[ix['Hips'],:3,3]-pr['origin'])@np.array(pr['camera'])).tolist()
  camera=dict(camera=pr['camera'],origin=pr['origin'],anchor_camera_xyz=anchor,projection='orthographic',inverse_distance=0,pixel_scale=pr['pixel_scale'],offset=pr['offset'],image_size=[535,687],embedded_in_fbx=False)
  (dest/'camera.json').write_text(json.dumps(camera,indent=2));met.update(geometry_accepted=False,rmse_px=met['old_rmse'],bound_hit_bones=[],collision_fallback=True)
  met['orientation_stage']=dict(met.get('orientation_stage',{}),applied=False,reason='collision_rollback');met['regional_acceptance']=[];met['rejection_reasons'].append('tested_region_collision_regression');met['lens']['selected']='orthographic';met['lens']['reason']='collision_fallback';met['lens'].update(inverse_distance=0,pixel_scale=pr['pixel_scale'],offset=pr['offset']);row.update(geometry_accepted=False,projection='orthographic',reasons=met['rejection_reasons'])
 (dest/'metrics.json').write_text(json.dumps(met,indent=2))
 sys.argv=['blender','--',str(dest.resolve())];runpy.run_path(str(Path(__file__).resolve().parents[1]/'refine_manual_004/export_blender.py'),run_name='__main__')
 audit.append(dict(case_id=cid,rollback=bool(increases) and not shadow(),would_rollback=bool(increases),tested_crossing_increases=increases,export=json.loads((dest/'export-check.json').read_text()),certified=False));(root/'audit.json').write_text(json.dumps(audit,indent=2));print('COLLISION SCREEN',cid,'shadow_violation' if increases and shadow() else 'fallback' if increases else 'pass',flush=True)
(root/'results.json').write_text(json.dumps(rows,indent=2))

(root/'status.json').write_text(json.dumps(dict(status='audited_experimental',certified=False,export_count=len(audit)),indent=2))
