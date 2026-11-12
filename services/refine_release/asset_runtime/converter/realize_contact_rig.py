"""Realize only the contact arm on a validated baseline rest rig.

The imported mesh is a discarded planning carrier, never a final skin source.
Keeping other edit-bones untouched avoids unrelated toe roll roundtrip drift.
"""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import bpy,numpy as np
    from mathutils import Matrix
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    from converter.pose_safety import descendants
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('baseline-body','proposal','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--side',choices=('Left','Right'),required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    import json
    with np.load(a.proposal,allow_pickle=False) as z:x={k:z[k] for k in z.files}
    names=list(x['names']);parents=json.loads(str(x['parents']));active=descendants(names,parents,'mixamorig:'+a.side+'Arm')
    rt.import_character=import_full_character;rt.reset_scene();arm,meshes=rt.import_character(str(a.baseline_body))
    if sorted(b.name for b in arm.data.bones)!=names:raise ValueError('contact_rig_names')
    bpy.context.view_layer.objects.active=arm;arm.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
    for i in active:
        bone=arm.data.edit_bones[names[i]];length=bone.length
        bone.matrix=arm.matrix_world.inverted()@Matrix(x['body_bones'][i]);bone.length=length
    bpy.ops.object.mode_set(mode='OBJECT');bpy.context.view_layer.update()
    a.output.parent.mkdir(parents=True,exist_ok=True);rt.export_fbx(str(a.output),embed_textures=False)
if __name__=='__main__':main()
