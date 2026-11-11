"""Bake approved authoring rules around the unchanged V3.2.5 solver.

A neutral FBX goes through the existing solver. Its world deformation matrices
are transported back to the source rig, where approved pose correctives run.
Only the final, corrected fixed FBX is passed to PNG/GLB generation.
"""
from __future__ import annotations
import importlib.util,json,shutil,hashlib
from pathlib import Path
from .shape_spec import validate_source

def import_full_character(path):
    """Approved shape FBXs have no synthetic leaf bones; retain their 52 real bones."""
    import bpy
    from mathutils import Matrix
    from . import retarget as rt
    before=set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path),ignore_leaf_bones=False,automatic_bone_orientation=False)
    added=[o for o in bpy.data.objects if o not in before]
    arms=[o for o in added if o.type=='ARMATURE'];meshes=[o for o in added if o.type=='MESH']
    if len(arms)!=1:raise ValueError('shape FBX must contain exactly one rig')
    arm=arms[0];arm.animation_data_clear()
    # Existing solver exports synthetic *_end children. Remove only these
    # unweighted markers, never genuine leaf bones such as Head/ToeBase.
    markers=[b.name for b in arm.data.bones if b.name.endswith('_end') and not b.children
             and not any(m.vertex_groups.get(b.name) for m in meshes)]
    if markers:
        bpy.context.view_layer.objects.active=arm;arm.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        for name in markers:arm.data.edit_bones.remove(arm.data.edit_bones[name])
        bpy.ops.object.mode_set(mode='OBJECT')
    for b in arm.pose.bones:b.matrix_basis=Matrix.Identity(4)
    for m in meshes:m.animation_data_clear()
    bpy.context.view_layer.update();rt._apply_object_transform(arm)
    return arm,meshes

def _load(recipe):
    import bpy
    # Re-hash the approved source inside the job process: the paths came from the
    # registry, but the bytes on disk are checked again right before Blender opens them.
    validate_source(recipe['rules'],{k:recipe[k] for k in ('blend','controller','blend_sha256','controller_sha256')})
    bpy.ops.wm.open_mainfile(filepath=recipe['blend'])
    # No embedded file scripts are auto-executed; only the pinned module is loaded.
    spec=importlib.util.spec_from_file_location('_standin_shape_controller',recipe['controller'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if getattr(module,'VERSION',None)!=recipe['rules']:raise ValueError('loaded controller VERSION does not match shape rules')
    module.register();module.set_controls(recipe['values'],correct=False)
    return module

def convert_shaped(*, recipe, bvh_path, character_fbx, out_path, **options):
    import bpy,numpy as np
    from mathutils import Matrix
    from . import retarget as rt
    from .convert import convert
    module=_load(recipe);mesh,rig=module.objects()
    neutral=module.export_fixed(str(Path(out_path).parent/'neutral'))
    manifest=json.loads((neutral/'manifest.json').read_text())
    # Capture imported FBX rest frames, including FBX bone-axis conversion.
    rt.reset_scene();arm,_=rt.import_character(str(neutral/'model.fbx'))
    rest={b.name:arm.matrix_world@b.matrix_local for b in arm.data.bones}
    solver_output=Path(out_path).with_name('solver.fbx')
    report=convert(bvh_path=bvh_path,character_fbx=str(neutral/'model.fbx'),out_path=str(solver_output),**options)
    if not report.ok:return report
    rt.reset_scene();posed,_=rt.import_character(str(solver_output))
    if set(rest)!=set(b.name for b in posed.data.bones):raise ValueError('solver changed shape rig bone set')
    deform={b.name:(posed.matrix_world@b.matrix_local)@rest[b.name].inverted() for b in posed.data.bones}
    module=_load(recipe);mesh,rig=module.objects();module.BUSY=True
    try:
        inv=rig.matrix_world.inverted()
        # Parents before children; update after each assignment for Blender dependencies.
        for bone in rig.data.bones:
            rig.pose.bones[bone.name].matrix=inv@deform[bone.name]@rig.matrix_world@bone.matrix_local
            bpy.context.view_layer.update()
        transfer_error=max((rig.matrix_world@rig.pose.bones[b.name].matrix.translation-
                            deform[b.name]@rig.matrix_world@b.head_local).length for b in rig.data.bones)
        if transfer_error>1e-5:raise ValueError('solver-to-author joint transfer exceeded 0.01 mm')
        module.correct_pose(check_intersections=True,contacts=True)
        posed_bundle=module.export_fixed(str(Path(out_path).parent/'corrected'),posed=True)
    finally:module.BUSY=False
    shutil.copyfile(posed_bundle/'model.fbx',out_path)
    final=json.loads((posed_bundle/'manifest.json').read_text())
    channels = {b.name: {'rotation_mode': b.rotation_mode,
                         'location': list(b.location), 'scale': list(b.scale),
                         'rotation_quaternion': list(b.rotation_quaternion),
                         'rotation_euler': list(b.rotation_euler),
                         'rotation_axis_angle': list(b.rotation_axis_angle)} for b in rig.pose.bones}
    lineage={
      'schema':'shape-bake.v1','import_policy':'preserve-real-terminal-bones-v1','rules':recipe['rules'],'asset':recipe['asset'],
      'blend_sha256':recipe['blend_sha256'],'controller_sha256':recipe['controller_sha256'],
      'shape_sha256':recipe['shape_sha256'],'controls':recipe['values'],
      'pose_transfer_max_joint_error_mm':transfer_error*1000,
      'neutral_fbx_sha256':manifest['fbx_sha256'],'final_fbx_sha256':final['fbx_sha256'],
      'roundtrip_max_error_mm':final['roundtrip_max_error_mm'],
      'roundtrip_joint_error_mm':final['roundtrip_joint_error_mm'],
      'roundtrip_weight_error':final['roundtrip_weight_error'],
      'pose_qa':final['pose_qa'],'contact_qa':final['contact_qa'],
      # Native authoring frames avoid amplifying FBX axis/float round-trip noise
      # when an independent verifier reconstructs the requested source surface.
      # This records pose only: expected geometry is recomputed by that verifier.
      'authoring_pose': {'schema': 'shape-authoring-pose.v1', 'space': 'armature-local',
                         'bones': final['pose'], 'channels': channels},
    }
    Path(out_path).with_suffix('.shape.json').write_text(json.dumps(lineage,ensure_ascii=False,indent=2))
    return report
