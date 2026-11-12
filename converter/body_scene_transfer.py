"""Opt-in, one-job solver-to-author scene handoff; no cross-job scene cache.

The pinned solver and author exporters are unchanged. Only the source-body
transfer FBX (which has no saved-file verifier) is replaced by explicit matrices.
Every candidate lives in a private directory; exceptions retry the legacy path.
"""
from __future__ import annotations

from dataclasses import dataclass
import builtins
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

FLAG = 'CONVERTER_EXPERIMENT_BODY_SCENE_TRANSFER'
VERSION = 'body-scene-transfer.v1'


def enabled():
    return os.getenv(FLAG, '0') == '1'


@dataclass(frozen=True)
class SolverScene:
    """World rest frames after rigged-rest bake; no Blender object outlives load."""
    frames: dict
    source_body: dict | None
    disk_exports_avoided: int
    disk_imports_avoided: int
    memory_rig_bytes: int = 0


def canonical_frames(path):
    """Keep Blender's FBX bone-axis normalization, without transferring mesh.

    The pinned codec still serializes/reloads the rig, in a bounded BytesIO.
    A direct matrix handoff was rejected: omitting codec normalization changed
    corner normals after the author correctives. Only module-local open hooks
    for one private sentinel are installed; every other file uses normal I/O.
    """
    import bpy
    from io_scene_fbx import encode_bin, parse_fbx
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    sentinel = str(Path(path).with_suffix('.memory-rig.fbx'))
    data = None
    absent = object()
    prior = {module: vars(module).get('open', absent) for module in (encode_bin, parse_fbx)}

    class Sink(io.BytesIO):
        def close(self):
            nonlocal data
            if not self.closed:
                data = self.getvalue()
                super().close()

    def memory_open(file, mode='r', *args, **kwargs):
        if str(file) != sentinel:
            return builtins.open(file, mode, *args, **kwargs)
        if mode == 'wb':
            return Sink()
        if mode == 'rb' and data is not None:
            return io.BytesIO(data)
        raise ValueError('unsupported memory rig access')

    try:
        encode_bin.open = parse_fbx.open = memory_open
        # Bind-pose cluster TransformLink matrices affect Blender bone import.
        # Keep each mesh object/modifier/transform so those matrices are emitted,
        # but omit its surface: the next stage only consumes bone frames.
        replacements = []
        try:
            for obj in list(bpy.context.scene.objects):
                if obj.type == 'MESH':
                    surface = obj.data
                    tiny = bpy.data.meshes.new('scene-transfer-binding-proxy')
                    tiny.from_pydata([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], [], [(0, 1, 2)])
                    replacements.append((obj, surface, tiny))
                    obj.data = tiny
            rt.export_fbx(sentinel, embed_textures=False, bake_anim=False)
        finally:
            for obj, surface, tiny in replacements:
                obj.data = surface
                bpy.data.meshes.remove(tiny)
        if data is None or not 0 < len(data) <= 2*1024*1024:
            raise ValueError('memory rig export failed or exceeded bound')
        rt.reset_scene()
        arm, meshes = import_full_character(sentinel)
        if any(len(m.data.vertices) != 3 for m in meshes):
            raise ValueError('memory rig unexpectedly contains a body surface')
        frames = {b.name: (arm.matrix_world @ b.matrix_local).copy() for b in arm.data.bones}
        return frames, len(data)
    finally:
        for module, value in prior.items():
            if value is absent:
                del module.open
            else:
                module.open = value
        # Blender normally clears this after a successful write. A failed codec
        # must not retain wrappers for objects discarded by the legacy retry.
        from io_scene_fbx.fbx_utils import ObjectWrapper
        ObjectWrapper.cache_clear()


def solver_scene(path, bvh_path, *, recipe, rest, options):
    import bpy
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    from converter.source_body import apply_source_body, reviewed_frames
    from converter.solved_state import digest

    # This is the same canonical solver FBX the legacy source-body stage reads.
    rt.reset_scene()
    arm, meshes = import_full_character(str(path))
    raw = Path(bvh_path).read_bytes()
    source = raw.decode('utf-8-sig', errors='replace')
    try:
        recognized = ('ROOT mixamorig:Hips' in source and rt._offset_delta_pose(source) is not None)
    except ValueError:
        recognized = False
    body = None
    if recognized and not options.get('force_exact_v324', False):
        src = rt.import_bvh(str(bvh_path), frame=options.get('frame', 0))
        authored = reviewed_frames(digest(raw), recipe['asset_sha256'],
                                  mirror=options.get('mirror', False))
        body = apply_source_body(src, arm, mirror=options.get('mirror', False), authored=authored)
        # Legacy discards the changed scene if it cannot transfer the body.
        if body['status'] != 'transferred':
            raise ValueError('source body scene was not transferred')
        bpy.data.objects.remove(src, do_unlink=True)
        rt.apply_output_mode(arm, meshes, 'rigged_rest')
    if any(abs(pb.matrix_basis[r][c] - (r == c)) > 1e-5
           for pb in arm.pose.bones for r in range(4) for c in range(4)):
        raise ValueError('solver scene is not rigged rest')
    memory_bytes = 0
    if body is not None:
        frames, memory_bytes = canonical_frames(path)
    else:
        frames = {b.name: (arm.matrix_world @ b.matrix_local).copy() for b in arm.data.bones}
    if set(frames) != set(rest) or any(not math.isfinite(v) for m in frames.values() for row in m for v in row):
        raise ValueError('solver scene changed shape rig identity')
    # Unrecognized BVHs already go straight from solver file to author frames.
    return SolverScene(frames, body, int(body is not None), int(body is not None), memory_bytes)


def _candidate(*, recipe, bvh_path, character_fbx, out_path, trace, checkpoint, **options):
    """Mirror the approved shape orchestrator, replacing only its transfer stage."""
    from mathutils import Matrix
    from converter import retarget as rt
    from converter.convert import convert
    from converter.prepared_body import restore
    from converter.shape_bake import _load, transfer_solver_pose

    started = time.monotonic()
    prepared, status = restore(recipe, Path(out_path).parent/'prepared-neutral')
    if prepared:
        neutral_fbx = prepared['path']
        manifest = {'fbx_sha256': prepared['fbx_sha256']}
        rest = {name: Matrix(matrix) for name, matrix in prepared['rest_frames'].items()}
    else:
        module = _load(recipe)
        neutral = module.export_fixed(str(Path(out_path).parent/'neutral'))
        manifest = json.loads((neutral/'manifest.json').read_text())
        neutral_fbx = neutral/'model.fbx'
        rt.reset_scene()
        arm, _ = rt.import_character(str(neutral_fbx))
        rest = {b.name: arm.matrix_world @ b.matrix_local for b in arm.data.bones}
    preparation = dict(status=status, seconds=time.monotonic()-started,
                       generation_skipped=prepared is not None, rest_import_skipped=prepared is not None)
    if prepared:
        preparation['key'] = prepared['key']
    checkpoint()
    solver_output = Path(out_path).with_name('solver.fbx')
    report = convert(bvh_path=bvh_path, character_fbx=str(neutral_fbx), out_path=str(solver_output), **options)
    if not report.ok:
        raise ValueError('solver rejected candidate')
    checkpoint()
    scene = solver_scene(solver_output, bvh_path, recipe=recipe, rest=rest, options=options)
    deform = {name: frame @ rest[name].inverted() for name, frame in scene.frames.items()}
    trace.update(disk_full_mesh_exports_avoided=scene.disk_exports_avoided,
                 disk_full_mesh_imports_avoided=scene.disk_imports_avoided,
                 memory_rig_roundtrips=int(scene.memory_rig_bytes > 0), memory_rig_bytes=scene.memory_rig_bytes)
    # All carried values are Python-owned matrices before author open_mainfile.
    module = _load(recipe)
    mesh, rig = module.objects()
    module.BUSY = True
    evaluated = module.evaluated
    try:
        transfer_error = transfer_solver_pose(rig, deform)
        module.correct_pose(check_intersections=True, contacts=True)
        if (scene.source_body or {}).get('reviewed_pose_frames', {}).get('smooth_waist'):
            from converter.source_body import smooth_reviewed_waist
            neutral = module.compute(recipe['values'])[0]
            surface = smooth_reviewed_waist(mesh, rig, neutral, evaluated(mesh))
            module.evaluated = lambda obj: surface if obj == mesh else evaluated(obj)
        checkpoint()
        bundle = module.export_fixed(str(Path(out_path).parent/'corrected'), posed=True)
    finally:
        module.evaluated = evaluated
        module.BUSY = False
    checkpoint()
    shutil.copyfile(bundle/'model.fbx', out_path)
    final = json.loads((bundle/'manifest.json').read_text())
    channels = {b.name: dict(rotation_mode=b.rotation_mode, location=list(b.location), scale=list(b.scale),
        rotation_quaternion=list(b.rotation_quaternion), rotation_euler=list(b.rotation_euler),
        rotation_axis_angle=list(b.rotation_axis_angle)) for b in rig.pose.bones}
    lineage = dict(schema='shape-bake.v1', import_policy='preserve-real-terminal-bones-v1',
        rules=recipe['rules'], asset=recipe['asset'], blend_sha256=recipe['blend_sha256'],
        controller_sha256=recipe['controller_sha256'], shape_sha256=recipe['shape_sha256'], controls=recipe['values'],
        pose_transfer_max_joint_error_mm=transfer_error*1000, neutral_fbx_sha256=manifest['fbx_sha256'],
        final_fbx_sha256=final['fbx_sha256'], neutral_preparation=preparation,
        roundtrip_max_error_mm=final['roundtrip_max_error_mm'], roundtrip_joint_error_mm=final['roundtrip_joint_error_mm'],
        roundtrip_weight_error=final['roundtrip_weight_error'], pose_qa=final['pose_qa'], contact_qa=final['contact_qa'],
        authoring_pose=dict(schema='shape-authoring-pose.v1', space='armature-local', bones=final['pose'], channels=channels))
    if scene.source_body is not None:
        lineage['source_body'] = scene.source_body
    Path(out_path).with_suffix('.shape.json').write_text(json.dumps(lineage, ensure_ascii=False, indent=2))
    return report


class SceneTransferReport:
    def __init__(self, report, trace):
        self.report, self.trace = report, trace

    def __getattr__(self, name):
        return getattr(self.report, name)

    def as_dict(self):
        return {**self.report.as_dict(), 'scene_transfer': self.trace}


def convert_shaped(*, recipe, out_path, **options):
    from converter.shape_bake import convert_shaped as legacy
    if not enabled() or options.get('force_exact_v324', False):
        return legacy(recipe=recipe, out_path=out_path, **options)
    started = time.monotonic()
    trace = dict(schema=VERSION, status='attempting', disk_full_mesh_imports_avoided=0,
                 disk_full_mesh_exports_avoided=0, memory_rig_roundtrips=0, memory_rig_bytes=0)
    try:
        budget = float(os.getenv('CONVERTER_EXPERIMENT_BODY_SCENE_SECONDS', '120'))
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('invalid scene budget')

        def checkpoint():
            if time.monotonic()-started >= budget:
                raise TimeoutError('scene transfer budget exhausted')

        with tempfile.TemporaryDirectory(prefix='body-scene-', dir=Path(out_path).parent) as temporary:
            candidate = Path(temporary)/'artifact.fbx'
            report = _candidate(recipe=recipe, out_path=str(candidate), trace=trace, checkpoint=checkpoint, **options)
            checkpoint()
            shutil.copyfile(candidate.with_suffix('.shape.json'), Path(out_path).with_suffix('.shape.json'))
            shutil.copyfile(candidate, out_path)
        trace['status'] = 'transferred'
    except Exception as exc:
        trace.update(status='legacy_recovery', reason=type(exc).__name__,
                     attempt_seconds=time.monotonic()-started)
        recovery = time.monotonic()
        try:
            report = legacy(recipe=recipe, out_path=out_path, **options)
        except Exception as recovery_error:
            trace.update(status='failed', recovery_reason=type(recovery_error).__name__)
            raise
        finally:
            trace['recovery_seconds'] = time.monotonic()-recovery
            trace['seconds'] = time.monotonic()-started
            Path(out_path).with_suffix('.scene-transfer.json').write_text(json.dumps(trace, indent=2))
        if not report.ok:
            trace['status'] = 'failed'
    trace['seconds'] = time.monotonic()-started
    Path(out_path).with_suffix('.scene-transfer.json').write_text(json.dumps(trace, indent=2))
    return SceneTransferReport(report, trace)
