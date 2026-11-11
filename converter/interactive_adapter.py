"""Opt-in post-frozen-body/pre-final-export boundary. No HTTP activation.

The frozen V3.2.5 solver and shaped adapter still write intermediate FBXs.
A verified body .blend can be reused for hand-only revisions, including baked
shape, without retargeting. Export and validation run in independent processes.
BFF retains ownership, quarantine, review, expiry and current-selection authority.
"""
from __future__ import annotations

from pathlib import Path
import json
import os

from converter.protocol import SOLVER_VERSION, SOLVER_MANIFEST_SHA256
from converter.solved_state import SolvedPoseState, canonical, decode, digest, validate_identity

VERSION = 'post-frozen-file-canonical-snapshot-v2'
# Approved #168 fixed defaults contain real terminal Head/ToeBase bones and no
# synthetic leaf children. The frozen legacy importer drops those real bones.
# Limit the existing full-character importer to these exact immutable assets;
# generic/unrecognized assets retain the frozen import path.
TERMINAL_BONE_ASSETS = {
    'male-default83-v0-7-6-p1': '085dd9c65a3a66035f7e0ce9e0a9171b9abb0e33dada92194676d92e00c8b026',
    'female-default80-v0-3-9-p1': '712ac75e58efd4d6ccded0b1e931590900015de023327f1320cf6d7d94026ffc',
}


def _run_body_job(job):
    from converter.worker import run_job
    if (job.get('shape_recipe') is not None
            or TERMINAL_BONE_ASSETS.get(job['character_id']) != job['character_sha256']):
        return run_job(job)
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    previous = rt.import_character
    rt.import_character = import_full_character
    try:
        # run_job independently re-hashes the source and frozen lineage before
        # opening it, and retains every solver/export/final-skin safety gate.
        return run_job(job)
    finally:
        rt.import_character = previous


def runtime_revision():
    root = Path(__file__).parent
    names = ['interactive_camera.py', 'interactive_worker.py', 'interactive_adapter.py', 'interactive_scene.py', 'interactive_export.py',
             'interactive_revision.py', 'interactive_variants.py', 'shared/interactive_pose_v1.mjs',
             'solved_state.py', 'hands.py', 'hand_geometry.py', 'hand_library.py', 'hand_contact.py',
             'hand_library.v1.json', 'hand_rig.py', 'framing.py', 'camera.py',
             'shape_bake.py', 'final_artifact.py', 'protocol.py', 'worker.py',
             'reviewed_library.py', 'reviewed_source_fingers.py', 'source_body.py', 'body_material.py']
    return digest(b''.join(canonical([n, digest((root/n).read_bytes())]) for n in names))


def checked_runtime():
    from converter.worker import _frozen_lineage, FrozenLineageError
    from converter.protocol import EXPECTED_BLENDER_VERSION, EXPECTED_BLENDER_BUILD_HASH
    import bpy
    _frozen_lineage()
    build = bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else bpy.app.build_hash
    if not bpy.app.version_string.startswith(EXPECTED_BLENDER_VERSION) or build != EXPECTED_BLENDER_BUILD_HASH:
        raise FrozenLineageError('snapshot runtime differs from frozen Blender')


def atomic_write(path, data):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError('immutable artifact already exists')
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    # Atomic no-replace publication, including a concurrent duplicate publisher.
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _save_scene(path):
    import bpy
    if Path(path).exists():
        raise ValueError('snapshot already exists')
    bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False, compress=True)


def body_binding(identity):
    # No cross-owner global cache. Hand-dependent refine must change refined BVH
    # or refineVersion, which invalidates this entire body/shape key.
    validate_identity(identity)
    return {k: identity[k] for k in ('ownerBinding', 'source', 'body', 'shape', 'mirror', 'solver')}


def shape_binding(recipe):
    return dict(sha256=recipe['shape_sha256'] if recipe else None,
        rulesVersion=recipe['rules'] if recipe else 'none',
        sourceSha256=digest(canonical({k: recipe[k] for k in
            ('asset', 'asset_sha256', 'blend_sha256', 'controller_sha256')})) if recipe else None,
        controlsSha256=digest(canonical(recipe['values'])) if recipe else None)


def prepare_body(job, identity, destination):
    """Blender child only: run existing worker, capture a private approved base.

    job is a load_job-validated, server-created full-body job with hands absent.
    This deliberately retains every old solver, shape and file-validation gate.
    """
    from converter.worker import load_job
    from converter.solved_state import validate_identity
    from converter.interactive_scene import capture
    import bpy
    checked_runtime(); validate_identity(identity)
    # Revalidate from the server-created job file instead of trusting arbitrary
    # client meshes or a claimed successful body report.
    job = load_job(job)
    if (job.get('hands') is not None or job.get('output_scope') != 'full'
            or job.get('preview_view') or job.get('preview_model') or job['thumbnail'] is not None
            or job.get('camera_rotation') is not None or job.get('retarget_cache')
            or job.get('force_exact_v324') or job.get('hand_contact_screen')):
        raise ValueError('body base requires unframed unrotated current-solver output without hands')
    binding = body_binding(identity)
    if (binding['source']['refinedBvhSha256'] or binding['source']['originalBvhSha256']) != digest(Path(job['bvh_path']).read_bytes()):
        raise ValueError('body source lineage mismatch')
    if binding['body']['characterSha256'] != job['character_sha256'] or binding['body']['characterId'] != job['character_id'] or binding['mirror'] != job['mirror']:
        raise ValueError('body character binding mismatch')
    recipe = job.get('shape_recipe')
    if binding['shape'] != shape_binding(recipe):
        raise ValueError('body shape binding mismatch')
    if binding['solver'] != {'version': SOLVER_VERSION, 'manifestSha256': SOLVER_MANIFEST_SHA256,
                             'adapterVersion': VERSION, 'runtimeSha256': runtime_revision()}:
        raise ValueError('body solver binding mismatch')
    report = _run_body_job(job)
    if report.get('ok') is not True or report.get('artifact_validation', {}).get('skin_verified') is not True:
        raise ValueError(f"body was not accepted and file validated: ok={report.get('ok')}; missing={report.get('missing_required')}; error={report.get('error_code')}")
    # run_job's final validator left the approved final body loaded, including
    # shape corrections. No separate mutable scene is shared with the exporter.
    model = capture()
    directory = Path(destination); directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory/'body.blend'; _save_scene(snapshot)
    metadata = dict(schema='interactive-body-base-v1', binding=binding,
        runtimeRevision=runtime_revision(), snapshotSha256=digest(snapshot.read_bytes()),
        # Frozen diagnostics may contain Infinity for an unmeasurable rejected
        # candidate. Preserve/hash that legacy report without admitting any
        # nonfinite value into the authoritative state/model serialization.
        model=model, bodyReportSha256=digest(json.dumps(report, sort_keys=True,
            separators=(',', ':'), ensure_ascii=True).encode()), initialFbxRequired=True)
    atomic_write(directory/'base.json', canonical(metadata))
    return metadata


def load_body(directory, identity):
    import bpy
    from converter.interactive_scene import capture, compare
    checked_runtime()
    directory = Path(directory)
    metadata = decode((directory/'base.json').read_bytes())
    snapshot = directory/'body.blend'
    if (metadata['schema'] != 'interactive-body-base-v1' or metadata['binding'] != body_binding(identity)
            or metadata['runtimeRevision'] != runtime_revision() or snapshot.is_symlink()
            or metadata['snapshotSha256'] != digest(snapshot.read_bytes())):
        raise ValueError('body snapshot binding/integrity mismatch')
    bpy.ops.wm.open_mainfile(filepath=str(snapshot), load_ui=False, use_scripts=False)
    compare(metadata['model'], capture(metadata['model']['coordinates']['originBlender']))
    return metadata


def solve_from_body(base_directory, identity, destination, *, checkpoint):
    """Restore approved body/shape; latest hand selection wins at each checkpoint.

    checkpoint is a worker-supplied revision/cancellation fence. It must raise
    before publication if a newer hand/refine/shape/scope selection exists.
    Returned packet is display authority only; no final file receipt is issued.
    """
    from converter.hand_library import load_library, normalize_request
    from converter.hands import solve_hands
    from converter.framing import crop_meshes
    from converter.interactive_scene import capture
    import bpy
    validate_identity(identity); checkpoint()
    if identity['camera']['baked']:
        raise ValueError('v1 packet camera is display-only; baked-camera requests unsupported')
    library = load_library()
    if identity['hands']['librarySha256'] != library.sha256 or identity['hands']['rulesVersion'] != 'hand-bake-v1':
        raise ValueError('hand library/rules binding mismatch')
    load_body(base_directory, identity); checkpoint()
    arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    wanted = normalize_request({side: identity['hands'][side] for side in ('left', 'right')})
    hand_report = solve_hands(arm, meshes, wanted) if wanted else None
    # Existing hand self-collision diagnostics are stricter for this new display
    # boundary: never call a failed hand safety result authoritative.
    if hand_report and hand_report['self_collision_free'] is not True:
        raise ValueError('hand self-collision gate failed')
    from converter.hand_contact import require_clear_selected_hands, selected_hand_screen_enabled
    if selected_hand_screen_enabled(identity['scope']):
        require_clear_selected_hands(meshes)
    checkpoint()
    crop_report = crop_meshes(arm, meshes, identity['scope'])
    directory = Path(destination); directory.mkdir(parents=True, exist_ok=False)
    # FBX normal import uses Blender's packed split-normal representation. A
    # second export can quantize it differently even with identical geometry.
    # Freeze the file ONCE, then establish the display authority from its import.
    # Export readiness still requires an independent fresh-process reimport and
    # the unchanged full correspondence gate. Never export the snapshot again.
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    origin = capture()['coordinates']['originBlender']
    staged = directory/'staged.fbx'
    rt.export_fbx(str(staged), embed_textures=False, bake_anim=False)
    checkpoint()
    rt.reset_scene(); import_full_character(str(staged))
    model = capture(origin)
    state = SolvedPoseState.create(identity, model)
    snapshot = directory/'solved.blend'; _save_scene(snapshot)
    atomic_write(directory/'state.canonical.json', state.canonical_bytes)
    manifest = dict(schema='interactive-snapshot-v1', stateHash=state.state_hash,
        runtimeRevision=runtime_revision(), snapshotSha256=digest(snapshot.read_bytes()),
        stagedFbxSha256=digest(staged.read_bytes()),
        handReportSha256=digest(canonical(hand_report)), cropReportSha256=digest(canonical(crop_report)))
    atomic_write(directory/'snapshot.json', canonical(manifest))
    checkpoint()
    # Publish last; an exception/restart can leave private partial files but no
    # authoritative packet. BFF still fences revision before accepting this.
    atomic_write(directory/'packet.json', json.dumps(state.packet(), separators=(',', ':')).encode())
    return state


def load_solved(directory):
    directory = Path(directory)
    state = SolvedPoseState((directory/'state.canonical.json').read_bytes())
    metadata = decode((directory/'snapshot.json').read_bytes())
    snapshot = directory/'solved.blend'
    staged = directory/'staged.fbx'
    if (metadata['schema'] != 'interactive-snapshot-v1' or metadata['stateHash'] != state.state_hash
            or metadata['runtimeRevision'] != runtime_revision() or snapshot.is_symlink()
            or state.payload['identity']['solver']['runtimeSha256'] != runtime_revision()
            or not staged.is_file() or staged.is_symlink() or metadata.get('stagedFbxSha256') != digest(staged.read_bytes())
            or metadata['snapshotSha256'] != digest(snapshot.read_bytes())):
        raise ValueError('solved snapshot integrity mismatch')
    return state, snapshot
