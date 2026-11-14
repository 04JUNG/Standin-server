"""Region correspondence errors, with fingertip weights as an explicit nail proxy."""
import argparse,importlib.util,json,sys
from pathlib import Path
import bpy,numpy as np
p=argparse.ArgumentParser();p.add_argument('--candidate');p.add_argument('--quality');p.add_argument('--reference');p.add_argument('--assets',required=True);p.add_argument('--gender',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(a.assets);sex=a.gender;stem='standin-body-0-8-0' if sex=='male' else 'standin-female-body-0-4-0';bpy.ops.wm.open_mainfile(filepath=str(root/'originals'/(stem+'.blend')));m=next(o for o in bpy.context.scene.objects if o.get('standin_studio_mesh'));arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');co=np.empty(len(m.data.vertices)*3);m.data.shape_keys.key_blocks['Basis'].data.foreach_get('co',co);co=co.reshape(-1,3);names={g.index:g.name for g in m.vertex_groups};regions={k:np.zeros(len(co),bool) for k in ['hands','fingertips_nail_proxy','face','feet','shoulder','elbow','wrist','hip','knee']}
for v in m.data.vertices:
 for k,tokens in [('hands',('Hand',)),('face',('Head',)),('feet',('Foot','Toe'))]:regions[k][v.index]=sum(g.weight for g in v.groups if any(t in names[g.group] for t in tokens))>.2
 regions['fingertips_nail_proxy'][v.index]=any('Hand' in names[g.group] and names[g.group].endswith('3') and g.weight>.05 for g in v.groups)
for k,short,radius in [('shoulder','Arm',.07),('elbow','ForeArm',.055),('wrist','Hand',.045),('hip','UpLeg',.085),('knee','Leg',.065)]:
 for side in ('Left','Right'):regions[k]|=np.linalg.norm(co-np.array(arm.data.bones['mixamorig:'+side+short].head_local),axis=1)<radius
candidate=Path(a.candidate) if a.candidate else root/'compact'/(sex+'-protected-3-articulated-local');ids=np.load(candidate/'source-indices.npy');qroot=Path(a.quality) if a.quality else root/('quality-'+sex+'-local');q=json.loads((qroot/'quality.json').read_text());cases=q['candidates'][candidate.name];report={'region_definition':'Original neutral bone weights / joint spheres; fixed source IDs. Fingertips are a conservative nail-support proxy, not a nail segmentation or thickness guarantee.','cases':{}}
for case,data in cases.items():
 out={}
 for pose in data['poses']:
  source=np.load((Path(a.reference) if a.reference else root/('quality-'+sex))/(case+'-'+pose+'.npz'))['co'];actual=np.load(qroot/(candidate.name+'-'+case+'-'+pose+'.npz'))['co'];d=np.linalg.norm(actual-source[ids],axis=1)*1000
  out[pose]={k:{'retained':int(mask[ids].sum()),'original':int(mask.sum()),'max_mm':float(d[mask[ids]].max()),'p95_mm':float(np.percentile(d[mask[ids]],95))} for k,mask in regions.items() if mask[ids].any()}
 report['cases'][case]=out
(root/('regions-'+sex+'.json')).write_text(json.dumps(report,indent=2)+'\n');print('REGIONS',sex,flush=True)
