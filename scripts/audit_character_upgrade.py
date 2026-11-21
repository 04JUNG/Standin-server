"""Read the active authoring rigs and approved FBXs without executing file scripts."""
import json,sys
from pathlib import Path
import bpy,numpy as np
ROOT=Path('C:/workspaces/Standin-server')
OUT=ROOT/'data/diagnostics/staging-character-upgrade-20261111'
rows=[]
for gender,version,fbx in [('male','standin-body-0-7-6','standin-male-v5.4-soft-shoulders'),('female','standin-female-body-0-3-9','standin-female-v5.1-raised-shoulders')]:
    for kind,path in [('active',OUT/'active/sources'/(version+'.blend')),('approved',Path('C:/Users/ASUS/Downloads')/(fbx+'.fbx'))]:
        if kind=='active':bpy.ops.wm.open_mainfile(filepath=str(path))
        else:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.fbx(filepath=str(path),ignore_leaf_bones=True,automatic_bone_orientation=False)
        arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
        mesh=max((o for o in bpy.context.scene.objects if o.type=='MESH'),key=lambda o:len(o.data.vertices))
        p=np.array([v.co[:] for v in mesh.data.vertices]);keys=mesh.data.shape_keys
        row={'gender':gender,'kind':kind,'vertices':len(p),'bounds':[p.min(0).tolist(),p.max(0).tolist()],
             'bones':[b.name for b in arm.data.bones], 'keys':[k.name for k in keys.key_blocks] if keys else [],
             'properties':{k:mesh[k] for k in mesh.keys() if k in ('landmarks','asset_identity','base_heads','asset_lock')},
             'bone_frames':{b.name:{'head':list(b.head_local),'tail':list(b.tail_local)} for b in arm.data.bones},
             'materials':[m.name for m in mesh.data.materials],
             'matrix_world':[list(row) for row in mesh.matrix_world]}
        rows.append(row);print('AUDIT',gender,kind,len(p),len(arm.data.bones),flush=True)
(OUT/'geometry-audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
