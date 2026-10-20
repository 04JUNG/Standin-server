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
    parser.add_argument('--scope', choices=['full', 'half', 'bust', 'shoulder_head', 'head'], default='full')
    parser.add_argument('--shape-registry', type=Path)
    parser.add_argument('--character-id')
    parser.add_argument('--control', action='append', default=[], help='slider key=value')
    parser.add_argument('--left-hand')
    parser.add_argument('--right-hand')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.out.mkdir(parents=True, exist_ok=True)
    artifact = args.out / 'model.fbx'
    selected_convert = convert
    recipe = None
    if args.shape_registry:
        if not args.character_id:
            parser.error('--shape-registry requires --character-id')
        import os
        from functools import partial
        from converter.shape_spec import load_registry, resolve_shape, digest_shape
        from converter.shape_bake import convert_shaped, import_full_character
        os.environ['CONVERTER_SHAPE_REGISTRY'] = str(args.shape_registry.resolve())
        registered = load_registry()['characters'][args.character_id]
        asset_sha = hashlib.sha256(args.character.read_bytes()).hexdigest()
        controls = {c['key']: c['default'] for c in registered['controls']}
        for item in args.control:
            key, value = item.split('=', 1)
            controls[key] = float(value)
        envelope = {'schemaVersion': 'body-shape.v1', 'controls': controls,
                    'shapeSha256': digest_shape(args.character_id, asset_sha, controls)}
        recipe = resolve_shape(args.character_id, asset_sha, envelope)
        # Match the existing worker policy, retaining actual terminal bones.
        rt.import_character = import_full_character
        selected_convert = partial(convert_shaped, recipe=recipe)
    elif args.control:
        parser.error('--control requires --shape-registry')
    report = selected_convert(bvh_path=str(args.bvh.resolve()), character_fbx=str(args.character.resolve()),
                     out_path=str(artifact.resolve()))
    result = {'converter_ok': report.ok, 'scope': args.scope,
              'missing_required': report.missing_required}
    # A rejected converter may leave an intermediate artifact. Never treat it as deliverable.
    if not report.ok:
        artifact.unlink(missing_ok=True)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2))
        raise RuntimeError('Converter rejected source/model; no fallback attempted')
    if args.left_hand or args.right_hand:
        hand_report = bake_hands(str(artifact), {'left': args.left_hand, 'right': args.right_hand})
        result['hand_body_bones_preserved'] = hand_report['body_bones_preserved']
        result['hand_self_collision_free'] = hand_report['self_collision_free']
        result['hand_presets'] = {'left': args.left_hand, 'right': args.right_hand}
    if recipe:
        shape = json.loads(artifact.with_suffix('.shape.json').read_text())
        result['shape_controls_match'] = shape['controls'] == recipe['values']
        result['shape_digest_matches'] = shape['shape_sha256'] == recipe['shape_sha256']
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
