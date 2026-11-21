"""Convert a v1 interactive packet fixture (BFF/Client tests) to the binary v2 encoding.

The logical state (identity, model, safety) is kept exactly; only the canonical
encoding changes, so stateHash / packetModelSha256 are recomputed with the
Converter implementation. Output is JSON with the v2 bytes as base64.

    python scripts/make_interactive_v2_fixture.py <v1-fixture.json> <out.json>
"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from converter.solved_state import SolvedPoseState  # noqa: E402
from converter.solved_state_v2 import MEDIA_TYPE, encode_model, encode_state  # noqa: E402

SOURCES = ['converter/solved_state.py', 'converter/solved_state_v2.py', 'converter/shared/interactive_pose_v2.mjs']


def main(source, destination):
    fixture = json.loads(Path(source).read_text())
    v1 = SolvedPoseState(fixture['packet']['canonicalState'].encode('ascii'))
    if v1.state_hash != fixture['packet']['stateHash']:
        raise SystemExit('input fixture stateHash does not match its canonicalState')
    payload = v1.payload
    # Keep the fixture's own safety (older fixtures carry handChecksPassed=true).
    head = {k: payload[k] for k in ('schema', 'identity', 'safety', 'dependencies')}
    state = SolvedPoseState(encode_state(head, encode_model(payload['model'])))
    if state.payload != payload:
        raise SystemExit('v2 fixture is not lossless')
    out = dict(
        evidence='Lossless v2 re-encoding of the v1 protocol fixture (same identity/model); Converter Python implementation.',
        converterCommit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        sourceSha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
        mediaType=MEDIA_TYPE, v1StateHash=v1.state_hash, packet=state.packet(),
        stateBase64=base64.b64encode(state.canonical_bytes).decode('ascii'))
    if 'receipt' in fixture:
        out['receipt'] = {**fixture['receipt'], 'stateHash': state.state_hash,
                          'packetModelSha256': state.model_sha256}
    Path(destination).write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:3])
