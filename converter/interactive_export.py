"""Fresh-process snapshot exporter and fresh-process FBX verifier.

Only the verifier emits a receipt. The display packet cannot mint one.
Run via export_bundle; CLI paths are server-authored local artifacts, never URLs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from converter.solved_state import RECEIPT_SCHEMA, canonical, digest


class Superseded(RuntimeError):
    pass


def _run_child(binary, args, checkpoint, timeout):
    command = [str(binary), '--background', '--factory-startup', '--disable-autoexec',
               '--python-exit-code', '1', '--python', str(Path(__file__).resolve()), '--', *args]
    # A file-backed log avoids filling a pipe during large FBX import/export.
    import tempfile
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while process.poll() is None:
                checkpoint()
                if time.monotonic()-started > timeout:
                    raise TimeoutError('interactive export stage timed out')
                time.sleep(0.05)
            checkpoint()
            if process.returncode:
                raise RuntimeError('interactive export child failed; no verified receipt')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()


def export_bundle(binary, bundle, destination, expected_state_hash, *, checkpoint, timeout=180):
    """Host-side runner: isolated export -> isolated reimport -> latest fence.

    A BFF caller must recheck ownership/review/quarantine/expiry/current selection
    when accepting and downloading. This function grants none of those rights.
    """
    from converter.interactive_adapter import load_solved, atomic_write
    from converter.solved_state import decode
    state, _ = load_solved(bundle)
    if state.state_hash != expected_state_hash:
        raise ValueError('requested state hash mismatch')
    checkpoint()
    directory = Path(destination); directory.mkdir(parents=True, exist_ok=False)
    common = ['--bundle', str(Path(bundle).resolve()), '--destination', str(directory.resolve()),
              '--state-hash', expected_state_hash]
    _run_child(binary, ['--stage', 'export', *common], checkpoint, timeout)
    _run_child(binary, ['--stage', 'verify', *common], checkpoint, timeout)
    receipt = decode((directory/'verified.canonical.json').read_bytes())
    identity = state.payload['identity']
    if (receipt.get('schema') != RECEIPT_SCHEMA or receipt.get('stateHash') != expected_state_hash
            or receipt.get('revision') != identity['selection']['revision']
            or receipt.get('artifactSha256') != digest((directory/'artifact.fbx').read_bytes())
            or receipt.get('identity') != identity or receipt.get('verified') is not True):
        raise ValueError('export receipt binding mismatch')
    checkpoint()
    # Only this final publication is consumed by an operation; a killed verifier
    # or a late hand change never leaves a public ready marker.
    atomic_write(directory/'receipt.json', json.dumps(receipt, separators=(',', ':')).encode())
    return receipt


def child(stage, bundle, destination, expected_state_hash):
    import bpy
    from converter import retarget as rt
    from converter.interactive_adapter import checked_runtime, load_solved, atomic_write, runtime_revision
    from converter.interactive_scene import capture, compare, POSITION_TOLERANCE_M, MATRIX_TOLERANCE, WEIGHT_TOLERANCE
    from converter.final_artifact import validate
    checked_runtime()
    state, snapshot = load_solved(bundle)
    if state.state_hash != expected_state_hash:
        raise ValueError('export state changed')
    payload = state.payload; model = payload['model']; identity = payload['identity']
    origin = model['coordinates']['originBlender']
    directory = Path(destination); artifact = directory/'artifact.fbx'
    if stage == 'export':
        if artifact.exists():
            raise ValueError('export destination already exists')
        bpy.ops.wm.open_mainfile(filepath=str(snapshot), load_ui=False, use_scripts=False)
        compare(model, capture(origin))
        # Exact immutable file which established this packet's display surface.
        # Re-exporting the .blend would quantize split normals a second time.
        shutil.copyfile(Path(bundle)/'staged.fbx', artifact)
        # Exporter is a consumer; it must not alter the authoritative state.
        compare(model, capture(origin))
    else:
        # Retain real terminal bones for shaped rigs. The policy used to import
        # the frozen body is part of its state, never silently trim Head/Toe.
        from converter.shape_bake import import_full_character
        rt.import_character = import_full_character
        artifact_sha = digest(artifact.read_bytes())
        skin = validate(artifact, identity['scope'], len(model['bones']))
        actual = capture(origin)
        errors = compare(model, actual)
        if digest(artifact.read_bytes()) != artifact_sha:
            raise ValueError('FBX changed during validation')
        receipt = dict(schema=RECEIPT_SCHEMA, stateHash=state.state_hash,
            revision=identity['selection']['revision'], operation=identity['selection']['operation'],
            identity=identity, verified=True, artifactSha256=artifact_sha,
            artifactBytes=artifact.stat().st_size, runtimeRevision=runtime_revision(),
            packetModelSha256=digest(canonical(model)), reimportModelSha256=digest(canonical(actual)),
            skin=skin, correspondence=dict(topology=True, jointIdentity=True, skinIdentity=True,
                **errors, positionToleranceM=POSITION_TOLERANCE_M,
                matrixTolerance=MATRIX_TOLERANCE, weightTolerance=WEIGHT_TOLERANCE),
            safety=payload['safety'], limitations=['initial-fbx-dependency', 'not-full-collision-certificate',
                'display-camera-not-csp-camera', 'bff-authorization-required'])
        atomic_write(directory/'verified.canonical.json', canonical(receipt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['export', 'verify'], required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--state-hash', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    child(args.stage, args.bundle, args.destination, args.state_hash)


if __name__ == '__main__':
    main()
