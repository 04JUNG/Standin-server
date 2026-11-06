"""Blender-only read-only extraction of source rig FK for P1 diagnostics."""
import bpy, json, sys
from pathlib import Path
r = Path(sys.argv[sys.argv.index('--') + 1])
rows = []
for source in json.loads((r / 'pilot12-base/records.json').read_text()):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=source['base_fbx'])
    arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE'))
    bones = []
    for b in arm.pose.bones:
        bones.append(dict(name=b.name.split(':')[-1], parent=b.parent.name.split(':')[-1] if b.parent else None, matrix=[list(x) for x in arm.matrix_world @ b.matrix], local_length=b.length))
    rows.append(dict(case_id=source['case_id'], base_fbx=source['base_fbx'], bones=bones))
(r / 'review-v2/base_rigs.json').write_text(json.dumps(rows, indent=2))
