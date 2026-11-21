"""Binary solved state (v2): lossless with v1, strict canonical form, fast reuse."""
import json
from pathlib import Path
import random
import struct
import subprocess

import numpy as np
import pytest

from converter import solved_state_v2 as v2
from converter.interactive_adapter import (BODY_MODEL_FILE, STATE_FILES, body_binding, load_solved,
    read_body_metadata, runtime_revision)
from converter.interactive_reuse import rebind_export, rebind_solved
from converter.solved_state import (RECEIPT_SCHEMA, V1_ENCODING, SolvedPoseState, canonical, digest,
    model_digest)
from tests.converter.test_interactive_state import identity, model

V2 = v2.ENCODING
MODULES = Path(__file__).resolve().parents[2]/'converter/shared'


def rich_model(vertices=600, bones=24, seed=7):
    """Several meshes, split normals, 1..4 sorted influences, negative zero and extremes."""
    rng = random.Random(seed)
    eye = [float(r == c) for r in range(4) for c in range(4)]
    bone_list = [dict(name=f'b{i:02d}', parent=i - 1, restLocal=[rng.uniform(-2, 2) for _ in range(16)],
                      poseLocal=eye[:], inverseBind=[rng.uniform(-1, 1) for _ in range(16)]) for i in range(bones)]
    meshes = []
    for index, count in enumerate((vertices, vertices // 3)):
        normals = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(count + 7)]
        weights = []
        for _ in range(count):
            ids = sorted(rng.sample(range(bones), rng.randint(1, 4)))
            raw = [rng.random() + .05 for _ in ids]; total = sum(raw)
            weights.append([[i, w / total] for i, w in zip(ids, raw)])
        positions = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(count)]
        positions[0] = [-0.0, 5e-324, 1.7976931348623157e308]
        meshes.append(dict(name=f'mesh_{index:03d}', positions=positions, normals=normals,
            triangles=[[rng.randrange(count) for _ in range(3)] for _ in range(count * 2)],
            normalTriangles=[[rng.randrange(count + 7) for _ in range(3)] for _ in range(count * 2)],
            weights=weights))
    value = model(); value.update(bones=bone_list, meshes=meshes)
    value['coordinates']['originBlender'] = [0.125, -0.0, 1.0000000000000002]
    return value


@pytest.mark.parametrize('make', [model, rich_model])
def test_v2_is_lossless_and_hashes_its_own_bytes(make):
    v1 = SolvedPoseState.create(identity(), make())
    two = SolvedPoseState.create(identity(), make(), encoding=V2)
    assert (v1.encoding, two.encoding) == (V1_ENCODING, V2) and two.is_binary and not v1.is_binary
    assert two.payload == v1.payload
    # Decoding v2 reproduces the exact v1 canonical bytes, including -0.0 and denormals.
    assert canonical(two.payload) == v1.canonical_bytes
    assert two.state_hash == digest(two.canonical_bytes)
    assert two.model_sha256 == digest(two.canonical_bytes[two._model_offset:])
    assert two.model_sha256 == model_digest(make(), V2)
    assert v1.model_sha256 == digest(canonical(make())) == model_digest(make())
    assert SolvedPoseState(two.canonical_bytes).state_hash == two.state_hash
    assert len(two.canonical_bytes) < len(v1.canonical_bytes)


def test_v2_packet_is_small_and_v1_packet_unchanged():
    v1 = SolvedPoseState.create(identity(), model())
    two = SolvedPoseState.create(identity(), model(), encoding=V2)
    assert list(v1.packet()) == ['schema', 'operation', 'revision', 'stateHash', 'previewState',
                                 'exportState', 'canonicalEncoding', 'canonicalState']
    assert v1.packet()['canonicalEncoding'] == 'typed-f64be-json-v1'
    packet = two.packet()
    assert 'canonicalState' not in packet and packet['canonicalEncoding'] == V2
    assert packet['stateBytes'] == len(two.canonical_bytes) and packet['modelSha256'] == two.model_sha256
    assert packet['stateHash'] == two.state_hash and packet['operation'] == 'op' and packet['revision'] == 1


def test_with_identity_reuses_model_bytes_and_isolates_payloads():
    two = SolvedPoseState.create(identity(), rich_model(), encoding=V2)
    changed = identity(); changed['selection'].update(operation='next', revision=2)
    other = two.with_identity(changed)
    assert other.canonical_bytes[other._model_offset:] == two.canonical_bytes[two._model_offset:]
    assert other.model_sha256 == two.model_sha256 and other.state_hash != two.state_hash
    assert other.identity['selection']['operation'] == 'next' and two.identity['selection']['operation'] == 'op'
    value = other.payload; value['model']['meshes'][0]['positions'][0][0] = 9.
    value['model']['coordinates']['originBlender'][0] = 9.; value['model']['capabilities']['localHands'] = True
    fresh = other.payload['model']
    assert fresh['meshes'][0]['positions'][0][0] == -0.0
    assert fresh['coordinates']['originBlender'][0] == 0.125 and fresh['capabilities'] == {'localHands': False}
    with pytest.raises(ValueError):
        other.model_view.meshes[0]['positions'][0, 0] = 1.  # read-only views over immutable bytes


def _mutate(data, offset, raw):
    data = bytearray(data); data[offset:offset + len(raw)] = raw
    return bytes(data)


def _sections(state):
    """Absolute byte offsets of each array section of the first mesh."""
    data = state.canonical_bytes; start = state._model_offset
    (size,) = struct.unpack_from('<I', data, start + 8)
    at = v2._align(start + 12 + size)
    out = {'bones': at}; at += len(state.model_view.meta['bones']) * 48 * 8
    mesh = state.model_view.meta['meshes'][0]
    for name, width, count in [('positions', 8, mesh['vertices'] * 3), ('normals', 8, mesh['normals'] * 3),
                               ('triangles', 4, mesh['triangles'] * 3), ('normalTriangles', 4, mesh['triangles'] * 3),
                               ('weightOffsets', 4, mesh['vertices'] + 1), ('weightBones', 2, mesh['influences']),
                               ('weightValues', 8, mesh['influences'])]:
        out[name] = at; at = v2._align(at + width * count)
    return out


def tampered_states():
    state = SolvedPoseState.create(identity(), rich_model(60, 6), encoding=V2)
    data = state.canonical_bytes; at = _sections(state)
    mesh = state.model_view.meta['meshes'][0]
    nan = struct.pack('<d', float('nan'))
    yield 'magic', b'X' + data[1:]
    yield 'trailing byte', data + b'\0'
    yield 'truncated', data[:-8]
    yield 'nan position', _mutate(data, at['positions'], nan)
    yield 'nan bone matrix', _mutate(data, at['bones'], nan)
    yield 'triangle out of range', _mutate(data, at['triangles'], struct.pack('<I', mesh['vertices']))
    yield 'normal triangle out of range', _mutate(data, at['normalTriangles'], struct.pack('<I', mesh['normals']))
    yield 'offset start', _mutate(data, at['weightOffsets'], struct.pack('<I', 1))
    yield 'bone id out of range', _mutate(data, at['weightBones'], struct.pack('<H', 6))
    yield 'weight above one', _mutate(data, at['weightValues'], struct.pack('<d', 1.5))
    yield 'weight not normalized', _mutate(data, at['weightValues'], struct.pack('<d', 1e-3))
    # Nonzero padding after the 2-byte bone id section (influences*2 is rarely 8-aligned).
    end = at['weightBones'] + 2 * mesh['influences']
    if end % 8:
        yield 'nonzero padding', _mutate(data, end, b'\1')
    spaced = data[12:12 + struct.unpack_from('<I', data, 8)[0]].replace(b',', b', ', 1)
    yield 'noncanonical header', (data[:8] + struct.pack('<I', len(spaced)) + spaced
                                   + b'\0' * (v2._align(12 + len(spaced)) - 12 - len(spaced))
                                   + data[state._model_offset:])
    certified = state.payload; certified['safety']['finalFileVerified'] = True
    head = {k: certified[k] for k in ('schema', 'identity', 'safety', 'dependencies')}
    yield 'certifies final export', v2.encode_state(head, data[state._model_offset:])


@pytest.mark.parametrize('name,data', list(tampered_states()))
def test_python_reader_rejects_noncanonical_or_invalid_bytes(name, data):
    with pytest.raises(ValueError):
        SolvedPoseState(data)


def test_unsorted_or_duplicate_joint_ids_are_not_canonical():
    bad = model(); bad['meshes'][0]['weights'] = [[[0, .5], [0, .5]]] * 3
    with pytest.raises(ValueError):
        SolvedPoseState.create(identity(), bad, encoding=V2)
    two_bones = model(); eye = two_bones['bones'][0]['restLocal']
    two_bones['bones'].append(dict(name='Spine', parent=0, restLocal=eye[:], poseLocal=eye[:], inverseBind=eye[:]))
    two_bones['meshes'][0]['weights'] = [[[1, .5], [0, .5]]] * 3
    with pytest.raises(ValueError, match='order'):
        SolvedPoseState.create(identity(), two_bones, encoding=V2)


def write_bundle(root, state):
    root.mkdir()
    (root/'solved.blend').write_bytes(b'private snapshot')
    (root/'staged.fbx').write_bytes(b'verified immutable fbx')
    (root/STATE_FILES[state.encoding]).write_bytes(state.canonical_bytes)
    (root/'snapshot.json').write_bytes(canonical(dict(schema='interactive-snapshot-v1',
        stateHash=state.state_hash, runtimeRevision=runtime_revision(),
        snapshotSha256=digest((root/'solved.blend').read_bytes()),
        stagedFbxSha256=digest((root/'staged.fbx').read_bytes()))))
    (root/'packet.json').write_text(json.dumps(state.packet()))
    return state


def test_binary_bundle_rebinds_without_reencoding_and_exports(tmp_path):
    old = write_bundle(tmp_path/'old', SolvedPoseState.create(identity(), rich_model(), encoding=V2))
    assert load_solved(tmp_path/'old')[0].state_hash == old.state_hash
    cached = tmp_path/'cache'; cached.mkdir()
    raw = (tmp_path/'old/staged.fbx').read_bytes(); (cached/'artifact.fbx').write_bytes(raw)
    (cached/'verified.canonical.json').write_bytes(canonical(dict(schema=RECEIPT_SCHEMA, verified=True,
        runtimeRevision=runtime_revision(), artifactSha256=digest(raw), artifactBytes=len(raw),
        packetModelSha256=old.model_sha256, identity=old.identity, stateHash=old.state_hash,
        revision=1, operation='op')))
    changed = identity(); changed['selection'].update(operation='new-op', revision=2)
    new = rebind_solved(tmp_path/'old', tmp_path/'new', changed)
    assert new.is_binary and not (tmp_path/'new/state.canonical.json').exists()
    assert json.loads((tmp_path/'new/packet.json').read_text())['canonicalEncoding'] == V2
    receipt = rebind_export(cached, tmp_path/'export', tmp_path/'new')
    assert receipt['stateHash'] == new.state_hash and receipt['packetModelSha256'] == old.model_sha256


def test_bundle_with_ambiguous_or_mislabelled_state_is_rejected(tmp_path):
    state = write_bundle(tmp_path/'both', SolvedPoseState.create(identity(), model(), encoding=V2))
    (tmp_path/'both/state.canonical.json').write_bytes(SolvedPoseState.create(identity(), model()).canonical_bytes)
    with pytest.raises(ValueError):
        load_solved(tmp_path/'both')
    write_bundle(tmp_path/'mislabelled', state)
    (tmp_path/'mislabelled/state.canonical.bin').rename(tmp_path/'mislabelled/state.canonical.json')
    with pytest.raises(ValueError):
        load_solved(tmp_path/'mislabelled')


def _body(directory, model_value, *, fmt):
    directory.mkdir(); (directory/'body.blend').write_bytes(b'body snapshot')
    binding = body_binding(identity())
    metadata = dict(binding=binding, runtimeRevision=runtime_revision(),
        snapshotSha256=digest(b'body snapshot'), bodyReportSha256='e'*64, initialFbxRequired=True)
    if fmt == 'v2':
        raw = v2.encode_model(model_value); (directory/BODY_MODEL_FILE).write_bytes(raw)
        metadata.update(schema='interactive-body-base-v2', modelSha256=v2.model_digest(raw))
    else:
        metadata.update(schema='interactive-body-base-v1', model=model_value)
    (directory/'base.json').write_bytes(canonical(metadata))
    return binding


@pytest.mark.parametrize('fmt', ['v1', 'v2'])
def test_body_base_readers_accept_both_formats(tmp_path, fmt):
    value = rich_model(80, 8)
    binding = _body(tmp_path/'base', value, fmt=fmt)
    light = read_body_metadata(tmp_path/'base', binding)
    assert light['model']['bones'] == value['bones']
    full = read_body_metadata(tmp_path/'base', binding, with_model=True)
    assert full['model'] == value
    other = body_binding(identity()); other['mirror'] = True
    with pytest.raises(ValueError):
        read_body_metadata(tmp_path/'base', other)


def test_body_base_v2_model_tamper_is_rejected(tmp_path):
    binding = _body(tmp_path/'base', rich_model(40, 4), fmt='v2')
    path = tmp_path/'base'/BODY_MODEL_FILE; raw = bytearray(path.read_bytes()); raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        read_body_metadata(tmp_path/'base', binding)


def test_body_base_format_switch(monkeypatch):
    from converter.interactive_adapter import body_base_format
    monkeypatch.delenv('CONVERTER_INTERACTIVE_BODY_FORMAT', raising=False)
    assert body_base_format() == 'interactive-body-base-v2'
    monkeypatch.setenv('CONVERTER_INTERACTIVE_BODY_FORMAT', 'v1')
    assert body_base_format() == 'interactive-body-base-v1'
    monkeypatch.setenv('CONVERTER_INTERACTIVE_BODY_FORMAT', 'json')
    with pytest.raises(ValueError):
        body_base_format()


def test_variant_atlas_refuses_binary_states():
    from converter.interactive_variants import build_table
    state = SolvedPoseState.create(identity(), model(), encoding=V2)
    with pytest.raises(ValueError, match='typed JSON'):
        build_table([(state, {})], registered_character_sha256={'c'*64})


def test_js_reader_matches_python_bytes_and_rejects_tampering(tmp_path):
    v1 = SolvedPoseState.create(identity(), rich_model())
    two = SolvedPoseState.create(identity(), rich_model(), encoding=V2)
    (tmp_path/'v1').write_bytes(v1.canonical_bytes); (tmp_path/'v2').write_bytes(two.canonical_bytes)
    bad = dict(tampered_states())
    for name in ('nan position', 'triangle out of range', 'weight not normalized', 'trailing byte',
                 'noncanonical header', 'certifies final export', 'bone id out of range'):
        (tmp_path/name.replace(' ', '-')).write_bytes(bad[name])
    names = [n.replace(' ', '-') for n in ('nan position', 'triangle out of range', 'weight not normalized',
             'trailing byte', 'noncanonical header', 'certifies final export', 'bone id out of range')]
    script = f'''import fs from 'node:fs'; import assert from 'node:assert/strict'; import {{createHash}} from 'node:crypto';
      import {{readSolvedState}} from {json.dumps((MODULES/'interactive_pose_v1.mjs').as_uri())};
      import {{readSolvedStateV2, toV1Model, triangleIndexJson}} from {json.dumps((MODULES/'interactive_pose_v2.mjs').as_uri())};
      const dir = process.argv[1], read = n => new Uint8Array(fs.readFileSync(dir + '/' + n));
      const hash = b => createHash('sha256').update(b).digest('hex');
      // Exact comparison: every number by its IEEE754 bits (keeps -0 and denormals), keys sorted.
      const bits = v => Array.isArray(v) ? '[' + v.map(bits).join(',') + ']'
        : v && typeof v === 'object' ? '{{' + Object.keys(v).sort().map(k => k + ':' + bits(v[k])).join(',') + '}}'
        : typeof v === 'number' ? Buffer.from(new Float64Array([v]).buffer).toString('hex') : JSON.stringify(v);
      const old = await readSolvedState(read('v1'), {json.dumps(v1.state_hash)});
      const raw = read('v2'); const s = await readSolvedStateV2(raw, {json.dumps(two.state_hash)});
      assert.equal(s.modelSha256, {json.dumps(two.model_sha256)});
      assert.equal(bits(toV1Model(s.model)), bits(old.model));
      assert(Object.is(s.model.meshes[0].positions[0], -0));
      assert.equal(bits(s.identity), bits(old.identity));
      assert.equal(bits(s.safety), bits(old.safety));
      assert.equal(triangleIndexJson(s.model.meshes[0].triangles), JSON.stringify(old.model.meshes[0].triangles));
      assert(Object.isFrozen(s.model.meshes[0]) && Object.isFrozen(s.identity.selection));
      // A pooled Node Buffer at a misaligned offset must decode (the reader copies to offset 0).
      const pooled = Buffer.concat([Buffer.alloc(3), Buffer.from(raw)]).subarray(3);
      assert.notEqual(pooled.byteOffset % 8, 0);
      assert.equal((await readSolvedStateV2(pooled, {json.dumps(two.state_hash)})).modelSha256, s.modelSha256);
      pooled[pooled.length - 1] ^= 1;
      raw[raw.length - 1] ^= 1;  // caller-owned buffer mutation cannot reach the decoded views
      assert.equal(s.model.meshes.at(-1).weightValues.at(-1), {json.dumps(two.model_view.meshes[-1]['weightValues'][-1].item())});
      await assert.rejects(readSolvedStateV2(read('v2'), '0'.repeat(64)), /hash/);
      for (const name of {json.dumps(names)}) {{
        const b = read(name);
        await assert.rejects(readSolvedStateV2(b, hash(b)), Error, name);
      }}'''
    subprocess.run(['node', '--input-type=module', '-e', script, str(tmp_path)], check=True)


def test_converter_image_allowlist_ships_binary_state_runtime(tmp_path):
    """The image COPYs files one by one; the API and Blender child must import v2."""
    import shlex
    import shutil
    import sys
    root = Path(__file__).resolve().parents[2]
    for line in (root/'Dockerfile.converter').read_text().splitlines():
        parts = shlex.split(line, comments=True) if line.lstrip().startswith('COPY ') else []
        if len(parts) == 3 and parts[1].startswith('converter/') and parts[2].startswith('/app/converter/'):
            target = tmp_path/parts[2].removeprefix('/app/')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/parts[1], target)
    assert (tmp_path/'converter/shared/interactive_pose_v2.mjs').is_file()
    subprocess.run([sys.executable, '-I', '-c',
        'import sys; sys.path.insert(0, sys.argv[1]); '
        'from converter.solved_state import SolvedPoseState; from converter import solved_state_v2; '
        'from converter.interactive_adapter import runtime_revision; runtime_revision()', str(tmp_path)],
        check=True, cwd=tmp_path, capture_output=True, text=True)


def test_large_body_scale_is_fast():
    """139k-vertex male body scale: decode stays far below the typed JSON cost."""
    import time
    rng = np.random.default_rng(3); V = 139_474; T = 278_944; B = 54
    eye = np.eye(4).ravel().tolist()
    bones = [dict(name=f'b{i}', parent=i - 1, restLocal=eye, poseLocal=eye, inverseBind=eye) for i in range(B)]
    weights = [[[int(b), 1.0]] for b in rng.integers(0, B, V)]
    value = model(); value.update(bones=bones, meshes=[dict(name='mesh_000', positions=rng.random((V, 3)).tolist(),
        normals=rng.random((V, 3)).tolist(), triangles=rng.integers(0, V, (T, 3)).tolist(),
        normalTriangles=rng.integers(0, V, (T, 3)).tolist(), weights=weights)])
    state = SolvedPoseState.create(identity(), value, encoding=V2)
    started = time.perf_counter(); SolvedPoseState(state.canonical_bytes)
    assert time.perf_counter() - started < 1.0
    assert len(state.canonical_bytes) < 16 * 1024 * 1024


def test_snapshot_store_v2_end_to_end_reuses_core_and_serves_binary(tmp_path, monkeypatch):
    """prepare (v2 body base) -> solve v2 -> rebind on appearance revision -> /model bytes -> variants 409."""
    pytest.importorskip('fcntl')
    import base64, threading, time
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from fastapi import HTTPException
    from converter_api import interactive as api
    from converter.solved_state import decode
    raw = b'unchanged source'
    resolved = SimpleNamespace(metadata=SimpleNamespace(sha256='c'*64, rig_profile='mixamo'), path=tmp_path/'body.fbx')
    runner = SimpleNamespace(settings=SimpleNamespace(blender_binary='unused'), _process_slots=threading.Semaphore(1))
    store = api.SnapshotStore(tmp_path/'private', SimpleNamespace(resolve=lambda cid: resolved), runner)
    monkeypatch.setattr(api, 'resolve_shape', lambda *a: None)
    seen_bones = []
    def camera(descriptor, value, *rest):
        seen_bones.append(value['bones']); return identity()['camera']
    monkeypatch.setattr(api, 'camera_for_model', camera)
    calls = []
    def child(stage, job, ip, base, destination, checkpoint, *encoding):
        checkpoint(); calls.append((stage, encoding)); current = decode(Path(ip).read_bytes())
        if stage == 'prepare':
            base.mkdir(parents=True); (base/'body.blend').write_bytes(b'base')
            body = v2.encode_model(model()); (base/BODY_MODEL_FILE).write_bytes(body)
            (base/'base.json').write_bytes(canonical(dict(schema='interactive-body-base-v2',
                binding=body_binding(current), runtimeRevision=runtime_revision(), snapshotSha256=digest(b'base'),
                modelSha256=v2.model_digest(body), bodyReportSha256='e'*64, initialFbxRequired=True)))
        else:
            # v1 calls keep the previous six-argument form (no encoding argument).
            write_bundle(destination, SolvedPoseState.create(current, model(), encoding=encoding[0] if encoding else V1_ENCODING))
    monkeypatch.setattr(store, '_child', child)
    selection = dict(schema='interactive-selection-v1', ownerBinding='a'*64, candidateId='c',
        body=dict(characterId='registered-fixture', assetSha256='c'*64), shape=None, shapeSource=None,
        source=dict(variant='base', originalBvhSha256=digest(raw), outputBvhSha256=digest(raw)),
        camera=dict(source_bvh_sha256=digest(raw)), hands=None, scope='full', mirror=False)
    payload = dict(selection=selection, operation='one', person='p', revision=1, bvhBase64=base64.b64encode(raw).decode(),
        expiresAt=datetime.fromtimestamp(time.time() + 600, timezone.utc).isoformat())
    first = store.solve(payload, V2)
    second = store.solve({**payload, 'operation': 'two', 'revision': 2}, V2)
    assert calls == [('prepare', ()), ('solve', (V2,))]  # the appearance revision reused the v2 core
    assert seen_bones and seen_bones[0] == model()['bones']
    for result in (first, second):
        assert result['packet']['canonicalEncoding'] == V2 and 'canonicalState' not in result['packet']
    served = store.read(second['snapshotId'], 'a'*64, 'model')
    assert v2.is_v2(served) and digest(served) == second['packet']['stateHash']
    assert SolvedPoseState(served).identity['selection']['operation'] == 'two'
    with pytest.raises(HTTPException) as unavailable:
        store.variants(second['snapshotId'], 'a'*64)
    assert unavailable.value.status_code == 409
    # The same selection requested as v1 is a separate bundle (no cross-encoding reuse).
    legacy = store.solve({**payload, 'operation': 'three', 'revision': 3})
    assert legacy['packet']['canonicalEncoding'] == V1_ENCODING and calls[-1] == ('solve', ())
