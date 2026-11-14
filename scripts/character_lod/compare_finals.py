"""Compare actual Converter final FBXs (not just authoring evaluations)."""
import argparse,json,sys
from pathlib import Path
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from converter.shape_bake import import_full_character
from converter import retarget as rt
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(a.assets);rows=json.loads((root/'benchmark/results.json').read_text());registry=json.loads((root/'delivery-package/shape-bakes.host.json').read_text());results=[]
def read(path):
 rt.reset_scene();arm,meshes=import_full_character(path);m=meshes[0];co=np.array([list(m.matrix_world@v.co) for v in m.data.vertices]);bones={b.name:np.array(arm.matrix_world@b.matrix_local) for b in arm.data.bones};names={g.index:g.name for g in m.vertex_groups};groups={name:np.array([sum(g.weight for g in v.groups if names[g.group]==name)>.2 for v in m.data.vertices]) for name in names.values()};return co,bones,groups
for sex in ('male','female'):
 for case in ('base','preset','custom'):
  for skin in ('default','flesh-guide'):
   pair={v:next(r for r in rows if r['gender']==sex and r['case']==case and r['skin']==skin and r['variant']==v and r['trial']==0) for v in ('original','candidate')};one,two=[read(Path(pair[v]['file'].replace('/work',str(root),1))) for v in ('original','candidate')];cid=pair['candidate']['character'];folder=Path(registry['sources'][registry['characters'][cid]['rules']]['blend']).parent;ids=np.load(folder/'source-indices.npy');d=np.linalg.norm(one[0][ids]-two[0],axis=1)*1000;result={'gender':sex,'case':case,'skin':skin,'retained_world_max_mm':float(d.max()),'retained_world_p99_mm':float(np.percentile(d,99)),'retained_world_mean_mm':float(d.mean()),'rig_head_max_mm':max(float(np.linalg.norm(one[1][name][:3,3]-two[1][name][:3,3])*1000) for name in one[1])};protected=np.isin(ids,np.load(folder/'protected-source.npy'));result['protected_max_mm']=float(d[protected].max());result['bone_regions']={name:{'max_mm':float(d[mask[ids]].max()),'p99_mm':float(np.percentile(d[mask[ids]],99))} for name,mask in one[2].items() if mask[ids].any()};result['additional_final_correspondence_screen_pass']=result['retained_world_p99_mm']<=.5 and result['protected_max_mm']<=.05;results.append(result);print('FINAL_PAIR',sex,case,skin,result['retained_world_max_mm'],sorted(result['bone_regions'].items(),key=lambda x:x[1]['max_mm'],reverse=True)[:4],flush=True)
(root/'benchmark/final-pairs.json').write_text(json.dumps(results,indent=2)+'\n')
