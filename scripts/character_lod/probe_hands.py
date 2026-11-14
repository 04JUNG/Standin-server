"""Early native replay of recorded real authoring channels + real hand baker.

This isolates the hand regression; it does not replace fresh full Converter QA.
"""
import argparse,json,sys
from pathlib import Path
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.character_lod.quality import load
from converter.hands import bake_hands
from converter.shape_bake import import_full_character
from scripts.character_lod.correspondence import load as load_correspondence
from converter import retarget as rt
rt.import_character=import_full_character  # production shape path preserves real terminal bones
p=argparse.ArgumentParser();p.add_argument('--originals',required=True);p.add_argument('--candidate',required=True);p.add_argument('--baseline-results',required=True);p.add_argument('--gender',required=True);p.add_argument('--out',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);folder=Path(a.candidate);mapping=load_correspondence(folder);protected=mapping.exact_mask(np.load(folder/'protected-source.npy'));rows=json.loads(Path(a.baseline_results).read_text());results=[]
def read(path):
 rt.reset_scene();arm,meshes=import_full_character(path);m=meshes[0];co=np.array([list(m.matrix_world@v.co) for v in m.data.vertices]);names={g.index:g.name for g in m.vertex_groups};hand=np.array([sum(g.weight for g in v.groups if 'Hand' in names[g.group])>.01 for v in m.data.vertices]);return co,hand
for case in ('base','preset','custom'):
 row=next(r for r in rows if r['gender']==a.gender and r['case']==case and r['variant']=='original' and r['skin']=='default' and r['trial']==0);files={};receipts={}
 for variant in ('original','candidate'):
  stem='standin-body-0-8-0' if a.gender=='male' else 'standin-female-body-0-4-0';blend=Path(a.originals)/(stem+'.blend') if variant=='original' else folder/'model.blend';controller=Path(a.originals)/(stem+'.py') if variant=='original' else folder/'controller.py';c=load(blend,controller);c.set_controls(row['controls'],correct=False);m,arm=c.objects();c.BUSY=True
  for name,channels in row['report']['shape_bake']['authoring_pose']['channels'].items():
   bone=arm.pose.bones[name]
   for key,value in channels.items():setattr(bone,key,value)
  arm.update_tag();bpy.context.view_layer.update();c.correct_pose();bundle=c.export_fixed(str(out/(case+'-'+variant)),posed=True);path=bundle/'model.fbx';receipts[variant]=bake_hands(path,{'left':'fist','right':'open'});files[variant]=str(path)
 original,hand=read(files['original']);candidate,_=read(files['candidate']);d=np.linalg.norm(mapping.apply(original)-candidate,axis=1)*1000;handmax=float(d[mapping.membership(hand)].max());result={'case':case,'files':files,'hand_max_mm':handmax,'protected_max_mm':float(d[protected].max()),'retained_p99_mm':float(np.percentile(d,99)),'retained_max_mm':float(d.max()),'root_repairs':{k:v['rig_repairs'] for k,v in receipts.items()},'pass':bool(handmax<=.05 and d[protected].max()<=.05 and np.percentile(d,99)<=.5)};results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print('HAND_REPLAY',a.gender,case,result,flush=True)
assert all(r['pass'] for r in results),'hand replay failed; do not proceed to promotion'
