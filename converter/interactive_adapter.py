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
from converter.solved_state import SolvedPoseState, canonical, decode, digest, model_digest, validate_identity

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
             'interactive_revision.py', 'interactive_reuse.py', 'interactive_variants.py', 'shared/interactive_pose_v1.mjs',
             'shared/interactive_pose_v2.mjs', 'solved_state.py', 'solved_state_v2.py',
             'hands.py', 'hand_geometry.py', 'hand_library.py', 'hand_contact.py',
             'hand_library.v1.json', 'hand_rig.py', 'framing.py', 'camera.py',
             'shape_bake.py', 'final_artifact.py', 'protocol.py', 'worker.py',
             'reviewed_library.py', 'reviewed_source_fingers.py', 'source_body.py',
             'reviewed_pose_frames.json', 'body_material.py']
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
    metadata = dict(binding=binding,
        runtimeRevision=runtime_revision(), snapshotSha256=digest(snapshot.read_bytes()),
        # Frozen diagnostics may contain Infinity for an unmeasurable rejected
        # candidate. Preserve/hash that legacy report without admitting any
        # nonfinite value into the authoritative state/model serialization.
        bodyReportSha256=digest(json.dumps(report, sort_keys=True,
            separators=(',', ':'), ensure_ascii=True).encode()), initialFbxRequired=True)
    if body_base_format() == BODY_BASE_V2:
        # The parent API process only needs bindings and camera bones. A small
        # JSON index plus a hash-pinned binary model avoids re-parsing the full
        # surface as typed JSON on every hand/appearance revision.
        from converter import solved_state_v2 as v2
        raw = v2.encode_model(model)
        atomic_write(directory/BODY_MODEL_FILE, raw)
        metadata = dict(metadata, schema=BODY_BASE_V2, modelSha256=v2.model_digest(raw))
    else:
        metadata = dict(metadata, schema=BODY_BASE_V1, model=model)
    atomic_write(directory/'base.json', canonical(metadata))
    return metadata


BODY_BASE_V1 = 'interactive-body-base-v1'
BODY_BASE_V2 = 'interactive-body-base-v2'
BODY_MODEL_FILE = 'base.model.bin'


def body_base_format():
    """Private body snapshot format. Readers accept both; rollback writes v1."""
    value = os.getenv('CONVERTER_INTERACTIVE_BODY_FORMAT', 'v2').strip().lower()
    if value not in ('v1', 'v2'):
        raise ValueError('CONVERTER_INTERACTIVE_BODY_FORMAT must be v1 or v2')
    return BODY_BASE_V2 if value == 'v2' else BODY_BASE_V1


def read_body_metadata(directory, binding, *, with_model=False):
    """Verify a private body base. Returns metadata with ``model``.

    v2 always includes ``model['bones']`` (camera calibration); the full v1-shaped
    model is decoded only when with_model is set (Blender scene correspondence).
    """
    directory = Path(directory)
    source, snapshot = directory/'base.json', directory/'body.blend'
    if source.is_symlink() or snapshot.is_symlink():
        raise ValueError('body snapshot binding/integrity mismatch')
    metadata = decode(source.read_bytes())
    if (metadata.get('schema') not in (BODY_BASE_V1, BODY_BASE_V2) or metadata.get('binding') != binding
            or metadata.get('runtimeRevision') != runtime_revision()
            or metadata.get('snapshotSha256') != digest(snapshot.read_bytes())):
        raise ValueError('body snapshot binding/integrity mismatch')
    if metadata['schema'] == BODY_BASE_V2:
        from converter import solved_state_v2 as v2
        path = directory/BODY_MODEL_FILE
        if path.is_symlink():
            raise ValueError('body snapshot binding/integrity mismatch')
        raw = path.read_bytes()
        if v2.model_digest(raw) != metadata.get('modelSha256'):
            raise ValueError('body snapshot binding/integrity mismatch')
        view = v2.decode_model(raw)
        metadata['model'] = view.to_dict() if with_model else {
            'bones': view.bones(), 'coordinates': dict(view.meta['coordinates'])}
    return metadata


def load_body(directory, identity):
    import bpy
    from converter.interactive_scene import capture, compare
    checked_runtime()
    directory = Path(directory)
    metadata = read_body_metadata(directory, body_binding(identity), with_model=True)
    snapshot = directory/'body.blend'
    bpy.ops.wm.open_mainfile(filepath=str(snapshot), load_ui=False, use_scripts=False)
    compare(metadata['model'], capture(metadata['model']['coordinates']['originBlender']))
    return metadata


STATE_FILES = {'typed-f64be-json-v1': 'state.canonical.json', 'standin-binary-f64le-v2': 'state.canonical.bin'}


def state_file(directory):
    """The single canonical state file of a solved bundle (v1 JSON or v2 binary)."""
    found = [Path(directory)/name for name in STATE_FILES.values() if (Path(directory)/name).exists()]
    if len(found) != 1 or found[0].is_symlink():
        raise ValueError('solved snapshot integrity mismatch')
    return found[0]


SOLVE_SCHEMA = 'interactive-solve-v1'


def solve_from_body(base_directory, identity, destination, *, checkpoint, encoding='typed-f64be-json-v1',
                    publish=True):
    """Restore approved body/shape; latest hand selection wins at each checkpoint.

    checkpoint is a worker-supplied revision/cancellation fence. It must raise
    before publication if a newer hand/refine/shape/scope selection exists.

    With publish=False (the serving path) this only freezes ``staged.fbx`` and a
    ``solve.json`` index. ``publish_solved`` then runs in a separate fresh
    Blender process: one reimport of the exact bytes yields both the display
    packet and the verified file receipt. publish=True keeps the previous
    single-process behaviour (in-process reimport, no receipt) for local tools.
    """
    from converter.hand_library import load_library, normalize_request
    from converter.hands import solve_hands
    from converter.framing import crop_meshes
    from converter.interactive_scene import capture, scene_origin
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
    # Hand choices still bake into the pose; quality screening is no longer a serving gate.
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
    # Same scene invariants and bit-identical hip pivot as capture(), without
    # traversing the whole surface just to read the origin (~3 s per solve).
    origin = scene_origin()
    staged = directory/'staged.fbx'
    rt.export_fbx(str(staged), embed_textures=False, bake_anim=False)
    checkpoint()
    solve = dict(schema=SOLVE_SCHEMA, runtimeRevision=runtime_revision(), originBlender=origin,
        stagedFbxSha256=digest(staged.read_bytes()),
        handReportSha256=digest(canonical(hand_report)), cropReportSha256=digest(canonical(crop_report)))
    atomic_write(directory/'solve.json', canonical(solve))
    if not publish:
        return solve
    rt.reset_scene(); import_full_character(str(staged))
    return _publish(directory, identity, solve, capture(origin), encoding, receipt=None)


def _publish(directory, identity, solve, model, encoding, *, receipt):
    """Write the state, snapshot index, optional verified receipt, then the packet last."""
    import shutil
    state = SolvedPoseState.create(identity, model, encoding=encoding)
    snapshot = directory/'solved.blend'; _save_scene(snapshot)
    atomic_write(directory/STATE_FILES[state.encoding], state.canonical_bytes)
    manifest = dict(schema='interactive-snapshot-v1', stateHash=state.state_hash,
        runtimeRevision=runtime_revision(), snapshotSha256=digest(snapshot.read_bytes()),
        stagedFbxSha256=solve['stagedFbxSha256'],
        handReportSha256=solve['handReportSha256'], cropReportSha256=solve['cropReportSha256'])
    atomic_write(directory/'snapshot.json', canonical(manifest))
    if receipt is not None:
        verified = directory/VERIFIED_DIR; verified.mkdir()
        shutil.copyfile(directory/'staged.fbx', verified/'artifact.fbx')
        if digest((verified/'artifact.fbx').read_bytes()) != solve['stagedFbxSha256']:
            raise ValueError('immutable FBX copy changed')
        atomic_write(verified/'verified.canonical.json', canonical(receipt(state)))
    # Publish last; an exception/restart can leave private partial files but no
    # authoritative packet. BFF still fences revision before accepting this.
    atomic_write(directory/'packet.json', json.dumps(state.packet(), separators=(',', ':')).encode())
    return state


VERIFIED_DIR = 'verified'


def publish_solved(directory, identity, *, encoding='typed-f64be-json-v1'):
    """Fresh Blender process only: the single reimport of the frozen ``staged.fbx``.

    The same import that passes the final-file structural validation is captured
    as the display packet, so what the user sees is exactly what the verified
    file reimports to. Only this verifier mints the receipt; ``export`` merely
    rebinds it under the current operation/revision fences.
    """
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    from converter.final_artifact import validate
    from converter.interactive_scene import capture
    from converter.interactive_export import verified_receipt
    checked_runtime(); validate_identity(identity)
    directory = Path(directory); staged = directory/'staged.fbx'
    solve = decode((directory/'solve.json').read_bytes())
    if (solve.get('schema') != SOLVE_SCHEMA or solve.get('runtimeRevision') != runtime_revision()
            or staged.is_symlink() or not staged.is_file() or digest(staged.read_bytes()) != solve['stagedFbxSha256']):
        raise ValueError('solved snapshot integrity mismatch')
    # Shaped/approved rigs keep their real terminal bones (same importer as the
    # previous in-process reimport and the standalone verifier).
    previous = rt.import_character
    rt.import_character = import_full_character
    try:
        skin = validate(staged, identity['scope'])
    finally:
        rt.import_character = previous
    model = capture(solve['originBlender'])
    if digest(staged.read_bytes()) != solve['stagedFbxSha256']:
        raise ValueError('FBX changed during validation')
    zero = dict(maxPositionErrorM=0.0, maxMatrixError=0.0, maxWeightError=0.0, maxNormalError=0.0)
    def receipt(state):
        # The packet IS this reimport, so both digests are the same model digest.
        model_sha = state.model_sha256 if state.is_binary else model_digest(model, state.encoding)
        return verified_receipt(state, artifact_sha=solve['stagedFbxSha256'],
            artifact_bytes=staged.stat().st_size, skin=skin, errors=zero,
            packet_model_sha256=model_sha, reimport_model_sha256=model_sha)
    return _publish(directory, identity, solve, model, encoding, receipt=receipt)


def load_solved(directory, *, verified_state=None):
    directory = Path(directory)
    path = state_file(directory)
    raw = path.read_bytes()
    if verified_state is not None and raw != verified_state.canonical_bytes:
        raise ValueError('solved snapshot state changed')
    state = verified_state if verified_state is not None else SolvedPoseState(raw)
    if STATE_FILES[state.encoding] != path.name:
        raise ValueError('solved snapshot integrity mismatch')
    metadata = decode((directory/'snapshot.json').read_bytes())
    snapshot = directory/'solved.blend'
    staged = directory/'staged.fbx'
    revision = runtime_revision()
    if (metadata['schema'] != 'interactive-snapshot-v1' or metadata['stateHash'] != state.state_hash
            or metadata['runtimeRevision'] != revision or snapshot.is_symlink()
            or state.identity['solver']['runtimeSha256'] != revision
            or not staged.is_file() or staged.is_symlink() or metadata.get('stagedFbxSha256') != digest(staged.read_bytes())
            or metadata['snapshotSha256'] != digest(snapshot.read_bytes())):
        raise ValueError('solved snapshot integrity mismatch')
    return state, snapshot
