"""Single reimport: the fresh publish verifier mints the receipt; export only rebinds it."""
import json
import threading
import time
from pathlib import Path

import pytest

from converter import solved_state_v2 as v2
from converter.interactive_adapter import VERIFIED_DIR, runtime_revision
from converter.interactive_export import LIMITATIONS, verified_receipt
from converter.interactive_reuse import rebind_solved
from converter.solved_state import RECEIPT_SCHEMA, V1_ENCODING, SolvedPoseState, canonical, digest
from tests.converter.test_interactive_state import identity, model
from tests.converter.test_solved_state_v2 import rich_model, write_bundle

ZERO = dict(maxPositionErrorM=0.0, maxMatrixError=0.0, maxWeightError=0.0, maxNormalError=0.0)


@pytest.mark.parametrize('encoding', [V1_ENCODING, v2.ENCODING])
def test_safety_is_available_without_decoding_the_surface(encoding):
    state = SolvedPoseState.create(identity(), model(), encoding=encoding)
    assert state.safety == {'bodyAccepted': True, 'handChecksPassed': False,
                            'surface': 'existing-adapter-checks-only', 'finalFileVerified': False}
    copy = state.safety; copy['finalFileVerified'] = True
    assert state.safety['finalFileVerified'] is False


@pytest.mark.parametrize('encoding', [V1_ENCODING, v2.ENCODING])
def test_receipt_keeps_the_verifier_schema(encoding):
    state = SolvedPoseState.create(identity(), model(), encoding=encoding)
    receipt = verified_receipt(state, artifact_sha='f'*64, artifact_bytes=10, skin={'scope': 'full'},
        errors=ZERO, packet_model_sha256=state.model_sha256, reimport_model_sha256=state.model_sha256)
    assert list(receipt) == ['schema', 'stateHash', 'revision', 'operation', 'identity', 'verified',
        'artifactSha256', 'artifactBytes', 'runtimeRevision', 'packetModelSha256', 'reimportModelSha256',
        'skin', 'correspondence', 'safety', 'limitations']
    assert receipt['schema'] == RECEIPT_SCHEMA and receipt['verified'] is True
    assert receipt['stateHash'] == state.state_hash and receipt['identity'] == state.identity
    assert receipt['packetModelSha256'] == state.model_sha256 == receipt['reimportModelSha256']
    assert receipt['limitations'] == LIMITATIONS and receipt['safety'] == state.safety
    assert receipt['correspondence']['topology'] is True and receipt['correspondence']['maxPositionErrorM'] == 0.0


def published_bundle(root, state):
    """What publish_solved leaves behind: bundle + verified/{artifact.fbx, verified.canonical.json}."""
    write_bundle(root, state)
    verified = root/VERIFIED_DIR; verified.mkdir()
    raw = (root/'staged.fbx').read_bytes(); (verified/'artifact.fbx').write_bytes(raw)
    (verified/'verified.canonical.json').write_bytes(canonical(verified_receipt(state, artifact_sha=digest(raw),
        artifact_bytes=len(raw), skin={'scope': 'full'}, errors=ZERO,
        packet_model_sha256=state.model_sha256, reimport_model_sha256=state.model_sha256)))
    return raw


def store_with_entry(tmp_path, state):
    pytest.importorskip('fcntl')
    from types import SimpleNamespace
    from converter_api import interactive as api
    runner = SimpleNamespace(settings=SimpleNamespace(blender_binary='unused'), _process_slots=threading.Semaphore(1))
    store = api.SnapshotStore(tmp_path/'private', None, runner)
    directory = tmp_path/'snapshot'; directory.mkdir()
    raw = published_bundle(directory/'solved', state)
    core = tmp_path/'core'; core.mkdir()
    store.entries['s'] = dict(owner='a'*64, person='p', revision=1, request='r', busy=False, directory=directory,
        expiry=time.monotonic() + 60, checkpoint=lambda: None, stateHash=state.state_hash, encoding=state.encoding,
        modelFile='solved/packet.json', packetSha256=digest((directory/'solved/packet.json').read_bytes()),
        runtimeRevision=runtime_revision(), core=core, base=dict(lock=threading.Lock()))
    return store, api, directory, raw


@pytest.mark.parametrize('encoding', [V1_ENCODING, v2.ENCODING])
def test_export_rebinds_the_published_receipt_without_running_the_verifier(tmp_path, monkeypatch, encoding):
    state = SolvedPoseState.create(identity(), rich_model(), encoding=encoding)
    store, api, directory, raw = store_with_entry(tmp_path, state)
    def forbidden(*a, **k): raise AssertionError('export must reuse the publish verifier receipt')
    monkeypatch.setattr(api, 'export_bundle', forbidden)
    receipt = store.export('s', 'a'*64)
    assert receipt['stateHash'] == state.state_hash and receipt['verified'] is True
    assert (directory/'export/artifact.fbx').read_bytes() == raw
    assert json.loads((directory/'export/receipt.json').read_bytes())['stateHash'] == state.state_hash
    assert store.read('s', 'a'*64, 'file') == raw


def test_tampered_published_receipt_falls_back_to_the_fresh_verifier(tmp_path, monkeypatch):
    state = SolvedPoseState.create(identity(), model(), encoding=v2.ENCODING)
    store, api, directory, raw = store_with_entry(tmp_path, state)
    (directory/'solved'/VERIFIED_DIR/'artifact.fbx').write_bytes(b'tampered')
    calls = []
    def verifier(binary, bundle, destination, state_hash, *, checkpoint):
        calls.append(state_hash); destination.mkdir()
        (destination/'artifact.fbx').write_bytes(raw)
        receipt = verified_receipt(state, artifact_sha=digest(raw), artifact_bytes=len(raw), skin={}, errors=ZERO,
            packet_model_sha256=state.model_sha256, reimport_model_sha256=state.model_sha256)
        (destination/'verified.canonical.json').write_bytes(canonical(receipt))
        (destination/'receipt.json').write_text(json.dumps(receipt))
        return receipt
    monkeypatch.setattr(api, 'export_bundle', verifier)
    assert store.export('s', 'a'*64)['stateHash'] == state.state_hash
    assert calls == [state.state_hash]


def test_core_reuse_carries_the_published_receipt_to_a_new_revision(tmp_path):
    from converter.interactive_reuse import rebind_export
    old = SolvedPoseState.create(identity(), rich_model(), encoding=v2.ENCODING)
    published_bundle(tmp_path/'core', old)
    changed = identity(); changed['selection'].update(operation='next', revision=2)
    new = rebind_solved(tmp_path/'core', tmp_path/'next', changed)
    assert (tmp_path/'next'/VERIFIED_DIR/'verified.canonical.json').is_file()
    receipt = rebind_export(tmp_path/'next'/VERIFIED_DIR, tmp_path/'export', tmp_path/'next')
    assert receipt['stateHash'] == new.state_hash and receipt['operation'] == 'next' and receipt['revision'] == 2
