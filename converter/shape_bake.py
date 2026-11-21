"""Bake approved authoring rules around the unchanged V3.2.5 solver.

A neutral FBX goes through the existing solver. Its world deformation matrices
are transported back to the source rig, where approved pose correctives run.
Only the final, corrected fixed FBX is passed to PNG/GLB generation.
"""
from __future__ import annotations
import importlib.util,json,shutil,hashlib,time
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


def transfer_solver_pose(rig, deform):
    """Transfer exact solver frames, then evaluate the approved shoulder girdle.

    FBX omits the drivers. Muting them during transfer prevents driven helper
    translations from being incorrectly counteracted by the pose assignment.
    Upper-arm orientation stays at the requested solver angle while its socket
    is allowed to rise with the clavicle/scapular rig.
    """
    import bpy
    helpers = {'Standin_LeftScapularLift', 'Standin_RightScapularLift'}
    approved = helpers.issubset(rig.data.bones.keys())
    drivers = list(rig.animation_data.drivers) if approved and rig.animation_data else []
    muted = [d.mute for d in drivers]
    inv = rig.matrix_world.inverted()
    desired = {b.name: inv @ deform[b.name] @ rig.matrix_world @ b.matrix_local for b in rig.data.bones}
    try:
        for d in drivers: d.mute = True
        for bone in rig.data.bones:
            rig.pose.bones[bone.name].matrix = desired[bone.name]
            bpy.context.view_layer.update()
        error = max((rig.matrix_world @ rig.pose.bones[b.name].matrix.translation -
                     deform[b.name] @ rig.matrix_world @ b.head_local).length for b in rig.data.bones)
        if error > 1e-5:
            raise ValueError('solver-to-author joint transfer exceeded 0.01 mm')
    finally:
        for d, was_muted in zip(drivers, muted): d.mute = was_muted
    if approved:
        driven = helpers | {'mixamorig:LeftShoulder', 'mixamorig:RightShoulder'}
        upper_arms = {'mixamorig:LeftArm', 'mixamorig:RightArm'}
        def put_rotation(name):
            bone = rig.data.bones[name]
            pb = rig.pose.bones[name]
            rest = bone.matrix_local.to_quaternion()
            if pb.parent:
                relative = pb.parent.bone.matrix_local.to_quaternion().inverted() @ rest
                rotation = relative.inverted() @ pb.parent.matrix.to_quaternion().inverted() @ desired[name].to_quaternion()
            else:
                rotation = rest.inverted() @ desired[name].to_quaternion()
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = rotation
            bpy.context.view_layer.update()
        for _ in range(24):
            bpy.context.view_layer.update()
            for name in sorted(upper_arms): put_rotation(name)
            if max(rig.pose.bones[n].matrix.to_quaternion().rotation_difference(desired[n].to_quaternion()).angle
                   for n in upper_arms) < .0005: break
        for bone in rig.data.bones:
            if bone.name not in driven | upper_arms: put_rotation(bone.name)
        angular_error = max(rig.pose.bones[n].matrix.to_quaternion().rotation_difference(desired[n].to_quaternion()).angle
                            for n in desired if n not in driven)
        if angular_error > .002:
            raise ValueError('approved shoulder transfer did not converge within 0.12 degrees')
    return error

def convert_shaped(*, recipe, bvh_path, character_fbx, out_path, **options):
    import bpy,numpy as np
    from mathutils import Matrix
    from . import retarget as rt
    from .convert import convert
    from .prepared_body import restore
    started=time.monotonic()
    prepared,preparation_status=restore(recipe,Path(out_path).parent/'prepared-neutral')
    if prepared:
        neutral_fbx=prepared['path']
        manifest={'fbx_sha256':prepared['fbx_sha256']}
        rest={name:Matrix(matrix) for name,matrix in prepared['rest_frames'].items()}
    else:
        module=_load(recipe);mesh,rig=module.objects()
        neutral=module.export_fixed(str(Path(out_path).parent/'neutral'))
        manifest=json.loads((neutral/'manifest.json').read_text())
        neutral_fbx=neutral/'model.fbx'
        # Capture imported FBX rest frames, including FBX bone-axis conversion.
        rt.reset_scene();arm,_=rt.import_character(str(neutral_fbx))
        rest={b.name:arm.matrix_world@b.matrix_local for b in arm.data.bones}
    preparation={'status':preparation_status,'seconds':time.monotonic()-started,
                 'generation_skipped':prepared is not None,'rest_import_skipped':prepared is not None}
    if prepared:preparation['key']=prepared['key']
    solver_output=Path(out_path).with_name('solver.fbx')
    report=convert(bvh_path=bvh_path,character_fbx=str(neutral_fbx),out_path=str(solver_output),**options)
    if not report.ok:return report
    source_body = None
    if not options.get('force_exact_v324', False):
        from converter.source_body import bake_source_body
        source_body = bake_source_body(solver_output, bvh_path,
            frame=options.get('frame', 0), mirror=options.get('mirror', False),
            character_sha256=recipe['asset_sha256'])
    rt.reset_scene();posed,_=rt.import_character(str(solver_output))
    if set(rest)!=set(b.name for b in posed.data.bones):raise ValueError('solver changed shape rig bone set')
    deform={b.name:(posed.matrix_world@b.matrix_local)@rest[b.name].inverted() for b in posed.data.bones}
    module=_load(recipe);mesh,rig=module.objects();module.BUSY=True
    evaluated = module.evaluated
    try:
        transfer_error=transfer_solver_pose(rig,deform)
        module.correct_pose(check_intersections=True,contacts=True)
        if (source_body or {}).get("reviewed_pose_frames", {}).get("smooth_waist"):
            from converter.source_body import smooth_reviewed_waist
            neutral = module.compute(recipe['values'])[0]
            waist_surface = smooth_reviewed_waist(mesh, rig, neutral, evaluated(mesh))
            # Use the corrected source surface in the pinned exporter's own
            # topology, weights and round-trip checks.
            module.evaluated = lambda obj: waist_surface if obj == mesh else evaluated(obj)
        posed_bundle=module.export_fixed(str(Path(out_path).parent/'corrected'),posed=True)
    finally:
        module.evaluated = evaluated
        module.BUSY = False
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
      'neutral_preparation':preparation,
      'roundtrip_max_error_mm':final['roundtrip_max_error_mm'],
      'roundtrip_joint_error_mm':final['roundtrip_joint_error_mm'],
      'roundtrip_weight_error':final['roundtrip_weight_error'],
      'pose_qa':final['pose_qa'],'contact_qa':final['contact_qa'],
      **({'source_body': source_body} if source_body is not None else {}),
      # Native authoring frames avoid amplifying FBX axis/float round-trip noise
      # when an independent verifier reconstructs the requested source surface.
      # This records pose only: expected geometry is recomputed by that verifier.
      'authoring_pose': {'schema': 'shape-authoring-pose.v1', 'space': 'armature-local',
                         'bones': final['pose'], 'channels': channels},
    }
    Path(out_path).with_suffix('.shape.json').write_text(json.dumps(lineage,ensure_ascii=False,indent=2))
    return report
