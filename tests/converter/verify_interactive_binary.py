"""Interactive preview path in real Blender 5.2, for both state encodings.

Synthetic CI fixtures only (make_fixtures.py). Runs the production children's
functions in-process: prepare_body (v2 and v1 body bases), solve_from_body as
typed JSON (v1) and binary (v2), then the fresh-process export_bundle verifier.
Checks that a real capture encodes losslessly in v2 (decodes to the exact v1
canonical bytes), that both encodings carry the same state, and that receipts
bind the state's own model digest.

blender --background --factory-startup --disable-autoexec --python-exit-code 1
    --python verify_interactive_binary.py -- <artifact-root>/assets <out>
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.protocol import JOB_SCHEMA_VERSION, SOLVER_VERSION, SOLVER_MANIFEST_SHA256  # noqa: E402
from converter.hand_library import load_library  # noqa: E402
from converter.interactive_adapter import (VERSION, BODY_MODEL_FILE, body_binding, load_solved, prepare_body,  # noqa: E402
    read_body_metadata, runtime_revision, shape_binding, solve_from_body)
from converter.interactive_export import export_bundle  # noqa: E402
from converter.interactive_scene import capture  # noqa: E402
from converter.solved_state import V1_ENCODING, SolvedPoseState, canonical  # noqa: E402
from converter import solved_state_v2 as v2  # noqa: E402

assets, out = map(Path, sys.argv[sys.argv.index('--') + 1:])
out.mkdir(parents=True, exist_ok=False)
character, bvh = assets/'character_mixamo.fbx', assets/'pose_raise.bvh'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence = dict(blender=os.environ.get('BLENDER_BINARY'), checks=[], timings={})


def check(name, ok):
    evidence['checks'].append(dict(name=name, passed=bool(ok)))
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    if not ok:
        raise AssertionError(name)


def timed(name, fn):
    started = time.monotonic(); value = fn(); evidence['timings'][name] = round(time.monotonic() - started, 3)
    return value


def job(name):
    directory = out/name; directory.mkdir()
    source = directory/'input.bvh'; source.write_bytes(bvh.read_bytes())
    value = dict(schema_version=JOB_SCHEMA_VERSION, conversion_id=name, temp_dir=str(directory),
        bvh_path=str(source), character_fbx=str(character), character_id='ci-synthetic-mixamo',
        character_sha256=sha(character), output_path=str(directory/'artifact.fbx'),
        report_path=str(directory/'report.json'), frame=0, mirror=False, output_mode='rigged_rest',
        apply_root_translation=False, embed_textures=False, force_exact_v324=False, thumbnail=None,
        output_scope='full')
    path = directory/'job.json'; path.write_text(json.dumps(value)); return path


eye = [float(r == c) for r in range(4) for c in range(4)]
identity = dict(ownerBinding='a'*64, selection=dict(operation='binary-verify', revision=1, person='0', candidate='fixture'),
    source=dict(originalBvhSha256=sha(bvh), refinedBvhSha256=None, refineVersion='none'),
    body=dict(characterId='ci-synthetic-mixamo', characterSha256=sha(character), rigProfile='mixamo'),
    shape=shape_binding(None),
    hands=dict(left=None, right=None, librarySha256=load_library().sha256, rulesVersion='hand-bake-v1'),
    scope='full', mirror=False,
    camera=dict(projection='orthographic', viewMatrix=eye, projectionMatrix=eye, distanceTorso=12.,
                torsoReferenceSha256=sha(character), baked=False),
    solver=dict(version=SOLVER_VERSION, manifestSha256=SOLVER_MANIFEST_SHA256, adapterVersion=VERSION,
                runtimeSha256=runtime_revision()))
binding = body_binding(identity)

# 1. Body base: default (v2) private format, and the v1 rollback format.
os.environ.pop('CONVERTER_INTERACTIVE_BODY_FORMAT', None)
base = timed('prepare_body_v2_base', lambda: prepare_body(job('body-v2'), identity, out/'base-v2'))
check('v2 body base writes small index + hash-pinned model', base['schema'] == 'interactive-body-base-v2'
      and 'model' not in base and (out/'base-v2'/BODY_MODEL_FILE).is_file())
light = timed('read_body_metadata_v2', lambda: read_body_metadata(out/'base-v2', binding))
full = read_body_metadata(out/'base-v2', binding, with_model=True)
check('API-side body metadata keeps camera bones', light['model']['bones'] == full['model']['bones'])
os.environ['CONVERTER_INTERACTIVE_BODY_FORMAT'] = 'v1'
legacy = timed('prepare_body_v1_base', lambda: prepare_body(job('body-v1'), identity, out/'base-v1'))
os.environ.pop('CONVERTER_INTERACTIVE_BODY_FORMAT')
check('v1 rollback base still embeds the model', legacy['schema'] == 'interactive-body-base-v1'
      and legacy['model'] == full['model'])

# 2. Solve the same selection as v1 and as v2 from the v2 base, and as v1 from the v1 base.
states = {}
for tag, encoding, source in [('v1', V1_ENCODING, 'base-v2'), ('v2', v2.ENCODING, 'base-v2'), ('v1-legacy-base', V1_ENCODING, 'base-v1')]:
    states[tag] = timed(f'solve_{tag}', lambda: solve_from_body(out/source, copy.deepcopy(identity), out/f'solved-{tag}',
                                                                 checkpoint=lambda: None, encoding=encoding))
    if tag == 'v2':
        # The scene now holds the reimported staged FBX the v2 state was captured from.
        model = capture(states['v2'].payload['model']['coordinates']['originBlender'])
        check('real capture: v2 encoding is deterministic',
              SolvedPoseState.create(identity, model, encoding=v2.ENCODING).canonical_bytes == states['v2'].canonical_bytes)
        check('real capture: v2 decodes to the exact v1 canonical bytes',
              canonical(states['v2'].payload) == SolvedPoseState.create(identity, model).canonical_bytes)
    loaded, _ = load_solved(out/f'solved-{tag}')
    check(f'{tag} bundle reloads with the same stateHash', loaded.state_hash == states[tag].state_hash)
s1, s2 = states['v1'], states['v2']
check('v2 state is binary and v1 is typed JSON', s2.is_binary and not s1.is_binary)
check('v1 and v2 solves carry the same logical state', s2.payload == s1.payload)
check('body base format does not change the solved v1 state', states['v1-legacy-base'].state_hash == s1.state_hash)
evidence['stateBytes'] = dict(v1=len(s1.canonical_bytes), v2=len(s2.canonical_bytes))
evidence['vertices'] = sum(len(m['positions']) for m in s1.payload['model']['meshes'])

# 3. Independent fresh-process FBX verifier on both encodings.
binary = os.environ.get('BLENDER_BINARY', '/opt/blender/blender')
for tag, state in (('v1', s1), ('v2', s2)):
    receipt = timed(f'export_bundle_{tag}', lambda: export_bundle(binary, out/f'solved-{tag}', out/f'export-{tag}',
                                                                  state.state_hash, checkpoint=lambda: None))
    check(f'{tag} export receipt is verified and bound', receipt['verified'] is True
          and receipt['stateHash'] == state.state_hash and receipt['packetModelSha256'] == state.model_sha256)
check('v2 receipt model digest is the binary model suffix hash',
      json.loads((out/'export-v2/receipt.json').read_text())['packetModelSha256']
      == v2.model_digest(s2.canonical_bytes, s2._model_offset))
check('v2 fresh-process reimport matched the binary packet (topology, joints, skin)',
      all(json.loads((out/'export-v2/receipt.json').read_text())['correspondence'][k]
          for k in ('topology', 'jointIdentity', 'skinIdentity')))

# 4. Serving path: solve only freezes staged.fbx; a fresh publish process reimports it once,
#    and its capture is both the display packet and the verified receipt.
import subprocess
from converter.interactive_adapter import VERIFIED_DIR
from converter.interactive_reuse import rebind_export
worker = Path(__file__).resolve().parents[2]/'converter/interactive_worker.py'
ip = out/'identity.json'; ip.write_bytes(canonical(identity))
for tag, encoding, reference in (('v1', V1_ENCODING, s1), ('v2', v2.ENCODING, s2)):
    destination = out/f'published-{tag}'
    solve = timed(f'serving_solve_{tag}', lambda: solve_from_body(out/'base-v2', copy.deepcopy(identity), destination,
                                                                  checkpoint=lambda: None, publish=False))
    check(f'{tag} serving solve freezes the FBX without publishing a packet',
          (destination/'staged.fbx').is_file() and not (destination/'packet.json').exists())
    check(f'{tag} solve origin equals the full-capture origin', solve['originBlender']
          == reference.payload['model']['coordinates']['originBlender'])
    timed(f'publish_child_{tag}', lambda: subprocess.run([binary, '--background', '--factory-startup', '--disable-autoexec',
        '--python-exit-code', '1', '--python', str(worker), '--', 'publish', 'unused', str(ip), 'unused',
        str(destination), encoding], check=True, capture_output=True))
    published, _ = load_solved(destination)
    check(f'{tag} published packet is byte-identical to the previous in-process reimport packet',
          published.canonical_bytes == reference.canonical_bytes)
    receipt = timed(f'export_published_{tag}', lambda: rebind_export(destination/VERIFIED_DIR,
                                                                    out/f'export-published-{tag}', destination))
    check(f'{tag} published receipt rebinds without another Blender process', receipt['verified'] is True
          and receipt['stateHash'] == published.state_hash and receipt['packetModelSha256'] == published.model_sha256)
    fallback = timed(f'export_fallback_{tag}', lambda: export_bundle(binary, destination, out/f'export-fallback-{tag}',
                                                                    published.state_hash, checkpoint=lambda: None))
    check(f'{tag} independent verifier still accepts the published bundle', fallback['verified'] is True
          and fallback['packetModelSha256'] == receipt['packetModelSha256'])

(out/'evidence.json').write_text(json.dumps(evidence, indent=2))
print(json.dumps(evidence, indent=2))
