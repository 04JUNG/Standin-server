"""Run inside Blender: --python this.py -- --bvh PRIVATE --character FBX --out TMP.

Uses existing Converter checks without fallback. Output contains aggregate geometry,
not source BVH, input images or credentials. Raw assets stay in the supplied out dir.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converter.convert import convert
from converter import retarget as rt
from converter.framing import process_output
from converter.hands import bake_hands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bvh', type=Path, required=True)
    parser.add_argument('--character', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scope', choices=['full', 'half', 'bust', 'head'], default='full')
    parser.add_argument('--left-hand')
    parser.add_argument('--right-hand')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.out.mkdir(parents=True, exist_ok=True)
    artifact = args.out / 'model.fbx'
    report = convert(bvh_path=str(args.bvh.resolve()), character_fbx=str(args.character.resolve()),
                     out_path=str(artifact.resolve()))
    result = {'converter_ok': report.ok, 'scope': args.scope,
              'missing_required': report.missing_required}
    # A rejected converter may leave an intermediate artifact. Never treat it as deliverable.
    if not report.ok:
        artifact.unlink(missing_ok=True)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2))
        raise RuntimeError('Converter rejected source/model; no fallback attempted')
    if args.left_hand or args.right_hand:
        bake_hands(str(artifact), {'left': args.left_hand, 'right': args.right_hand})
    result['framing'] = process_output(str(artifact), args.scope,
                                      preview_path=str(args.out / 'preview.png'))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    rt.reset_scene()
    arm, meshes = rt.import_character(str(artifact))
    points = [tuple(obj.matrix_world @ vertex.co) for obj in meshes for vertex in obj.data.vertices]
    result.update(meshes=len(meshes), vertices=len(points), bones=len(arm.data.bones),
                  finite=all(math.isfinite(x) for point in points for x in point),
                  bytes=artifact.stat().st_size,
                  reopened_same_bytes=digest == hashlib.sha256(artifact.read_bytes()).hexdigest())
    (args.out / 'result.json').write_text(json.dumps(result, indent=2))
    if not points or not result['finite'] or not result['reopened_same_bytes']:
        raise RuntimeError('Reopened geometry or artifact identity invalid')


if __name__ == '__main__':
    main()
