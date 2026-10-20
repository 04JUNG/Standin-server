"""Independently compare the app's full neutral mesh against delivered FBX vertices."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from converter import retarget as rt
from converter.shape_bake import import_full_character
R=ROOT/'artifacts/shape-staging';rows=[]
for p in sorted((R/'release/fbx').glob('*.fbx')):
 rt.reset_scene();a,meshes=import_full_character(str(p));assert len(a.data.bones)==52
 mesh=meshes[0];fbx=np.array([mesh.matrix_world@v.co for v in mesh.data.vertices]);app=np.fromfile(R/'runtime-qa'/(p.stem+'.f32'),dtype='<f4').reshape(-1,3)
 assert app.shape==fbx.shape
 err=float(np.linalg.norm(app-fbx,axis=1).max()*1000)
 assert err<.05,(p.stem,err)
 rows.append({'id':p.stem,'vertices':len(fbx),'max_error_mm':err});print(p.stem,err,flush=True)
(R/'runtime-qa/summary.json').write_text(json.dumps({'passed':True,'models':len(rows),'max_error_mm':max(r['max_error_mm'] for r in rows),'rows':rows},indent=2))
