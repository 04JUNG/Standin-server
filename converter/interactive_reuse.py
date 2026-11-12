"""Private, owner-bound geometry reuse with fresh operation authority.

Only validated server-authored bundles may be rebound. Appearance is absent
from this key; selected hands, source, rig, shape and camera remain included.
"""
from pathlib import Path
import copy
import json
import shutil
from converter.solved_state import canonical, digest, decode


def core_binding(identity):
    value = copy.deepcopy(identity)
    value['selection'] = {k: value['selection'][k] for k in ('person', 'candidate')}
    return value


def rebind_solved(source, destination, identity):
    from converter.interactive_adapter import load_solved, atomic_write, STATE_FILES
    old, _ = load_solved(source)
    if core_binding(old.identity) != core_binding(identity):
        raise ValueError('core geometry binding changed')
    # v2 rewrites only the small state header and reuses the validated model
    # bytes; v1 re-canonicalizes the full typed JSON as before.
    state = old.with_identity(identity)
    target = Path(destination)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("verified-export"))
    metadata = decode((target/'snapshot.json').read_bytes())
    metadata['stateHash'] = state.state_hash
    name = STATE_FILES[state.encoding]
    (target/name).unlink()
    atomic_write(target/name, state.canonical_bytes)
    (target/'snapshot.json').unlink()
    atomic_write(target/'snapshot.json', canonical(metadata))
    (target/'packet.json').unlink()
    atomic_write(target/'packet.json', json.dumps(state.packet(), separators=(',', ':')).encode())
    load_solved(target, verified_state=state)
    return state


def rebind_export(source, destination, bundle):
    from converter.interactive_adapter import load_solved, atomic_write, runtime_revision
    state, _ = load_solved(bundle)
    identity = state.identity
    previous = decode((Path(source)/'verified.canonical.json').read_bytes())
    raw = (Path(source)/'artifact.fbx').read_bytes()
    if (previous.get('verified') is not True or previous.get('runtimeRevision') != runtime_revision()
            or previous.get('artifactSha256') != digest(raw)
            or previous.get('artifactBytes') != len(raw)
            or previous.get('packetModelSha256') != state.model_sha256
            or core_binding(previous['identity']) != core_binding(identity)):
        raise ValueError('verified core artifact changed')
    receipt = {**previous, 'stateHash': state.state_hash,
        'revision': identity['selection']['revision'],
        'operation': identity['selection']['operation'],
        'identity': identity}
    target = Path(destination); target.mkdir(parents=True, exist_ok=False)
    atomic_write(target/'artifact.fbx', raw)
    atomic_write(target/'verified.canonical.json', canonical(receipt))
    atomic_write(target/'receipt.json', json.dumps(receipt, separators=(',', ':')).encode())
    return receipt
