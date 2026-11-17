"""Real Blender regression: neutral means no finger articulation, before FBX."""
import argparse,json,sys
from pathlib import Path
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.character_lod.quality import load,pose
p=argparse.ArgumentParser();p.add_argument('--blend',required=True);p.add_argument('--controller',required=True);p.add_argument('--out',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);c=load(a.blend,a.controller);m,arm=c.objects();controls=c.normalize(c.defaults(m));c.set_controls(controls,correct=False);expected=c.compute(controls)[0];actual=pose(c,'neutral');names={g.index:g.name for g in m.vertex_groups};hand=np.array([sum(g.weight for g in v.groups if 'Hand' in names[g.group])>.2 for v in m.data.vertices]);matrices={b.name:float(np.max(abs(np.array(b.matrix_basis)-np.eye(4)))) for b in arm.pose.bones if 'Hand' in b.name and any(x in b.name for x in ('Thumb','Index','Middle','Ring','Pinky'))};delta=float(np.linalg.norm(actual[hand]-expected[hand],axis=1).max()*1000);result={'neutral_finger_matrix_max_delta':max(matrices.values()),'neutral_hand_surface_max_mm':delta,'hand_vertices':int(hand.sum()),'passed':max(matrices.values())<1e-8 and delta<.01};Path(a.out).write_text(json.dumps(result,indent=2)+'\n');assert result['passed'],result;print('NEUTRAL_HAND_PASS',result,flush=True)
