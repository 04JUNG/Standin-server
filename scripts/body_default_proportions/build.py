"""Build review-only defaults from an immutable, approved shape release.

Run in the pinned Blender runtime; this never changes a source or activates a
catalog. Existing character IDs, recipes and asset bytes retain their meaning.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from converter.protocol import EXPECTED_BLENDER_BUILD_HASH, EXPECTED_BLENDER_VERSION
from converter.shape_spec import digest_shape, load_registry
from scripts.build_shape_release import load_controller


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def invariants(c):
    m, a = c.objects()
    # System keys and warped output rest positions intentionally follow controls.
    # The locked Basis, authored body targets and frozen rest must never change.
    c.source()
    uv = [(u.name, [list(v.uv) for v in u.data]) for u in m.data.uv_layers]
    return {
        'asset_lock': json.loads(m['asset_lock']),
        'frozen_rest_sha256': c.digest(json.loads(a['frozen_rest'])),
        'uv_sha256': c.digest(uv),
        'materials': [x.name if x else None for x in m.data.materials],
        'bones': [(b.name, b.parent.name if b.parent else None) for b in a.data.bones],
        'body_keys': sorted(k.name for k in m.data.shape_keys.key_blocks if k.name.startswith('BODY_')),
        # Calibration contains current warped rest matrices and is expected to
        # get a new hash. Preserve the policy and let its unchanged geometric
        # calibration gates reject unsupported rigs at every synchronize.
        'finger_policy': {k: json.loads(a['standin_finger_limits'])[k]
                          for k in ('version', 'source', 'neutral_preserved', 'coordinates')},
    }


def measure(c, values, coords):
    """World-Z stature / crown-to-chin height in neutral T pose.

    Use the authored chin plane and an actual vertex on that plane. A topology
    revision can invalidate stored vertex numbers, so verify them, then remap by
    the authored plane (nearest Z, central face, in front of head_y).
    """
    m, _ = c.objects()
    s = c.source()
    lm = json.loads(m['landmarks'])
    co = s['co']
    index = int(lm['chin_vertex'])
    remapped = bool(abs(co[index, 2] - lm['chin']) > 1e-5)
    if remapped:
        eligible = np.flatnonzero((abs(co[:, 0]) < .015) & (co[:, 1] < lm['head_y']))
        index = int(eligible[np.argmin(abs(co[eligible, 2] - lm['chin']))])
    if abs(co[index, 2] - lm['chin']) > 1e-5:
        raise ValueError('no actual chin vertex on authored plane')
    crown = int(np.argmax(coords[:, 2]))
    floor = float(coords[:, 2].min())
    head = float(coords[crown, 2] - coords[index, 2])
    height = float(coords[crown, 2] - floor)
    plane = float(c.warp([[0, lm['head_y'], lm['chin']]], values, m)[0, 2])
    return {
        'pose': 'T', 'axis': 'world Z; feet on floor',
        'definition': 'stature / (actual crown vertex Z - authored chin-plane vertex Z)',
        'heightMm': height * 1000, 'headHeightMm': head * 1000,
        'measuredHeads': height / head, 'requestedHeads': values['heads'],
        'errorHeads': height / head - values['heads'],
        'chinVertex': index, 'crownVertex': crown, 'storedChinIndexRemapped': remapped,
        'chinPlaneAgreementMm': float(abs(float(coords[index, 2]) - plane) * 1000),
        'soleZMm': floor * 1000,
    }


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--source-release', type=Path, required=True)
    ap.add_argument('--legacy-registry', type=Path, required=True)
    ap.add_argument('--catalog-package', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if bpy.app.version_string != EXPECTED_BLENDER_VERSION + ' LTS' or bpy.app.build_hash.decode() != EXPECTED_BLENDER_BUILD_HASH:
        raise ValueError('unapproved Blender build')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    registry = load_registry(args.source_release / 'shape-bakes.local.json')
    legacy = load_registry(args.legacy_registry)
    for section in ('sources', 'characters'):
        if set(registry[section]) & set(legacy[section]):
            raise ValueError('source releases must have disjoint immutable identities')
        registry[section].update(legacy[section])
    catalog = json.loads((args.catalog_package / 'body-models.json').read_text())
    chars = json.loads((args.catalog_package / 'characters.json').read_text())
    packaged = json.loads((args.catalog_package / 'shape-bakes.json').read_text())
    if set(registry['characters']) != set(packaged['characters']):
        raise ValueError('authoring character set differs from the active package')
    for cid, recipe in registry['characters'].items():
        active = packaged['characters'][cid]
        if any(recipe[k] != active[k] for k in ('rules', 'defaults', 'asset', 'preset')):
            raise ValueError('authoring recipe differs from the active package')
    # Earlier registration exports have different FBX bytes and control ordering.
    # Preserve the actual published recipes, never the preliminary export hashes.
    registry['characters'] = copy.deepcopy(packaged['characters'])
    # Local package paths are deployment paths; get immutable authoring sources
    # from the separately validated source release, never execute /app paths.
    recipes = copy.deepcopy(registry)
    base_catalog = copy.deepcopy(catalog)
    source_hashes = {}
    for source in recipes['sources'].values():
        for key in ('blend', 'controller'):
            src = Path(source[key])
            dst = out / 'sources' / src.name
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(src, dst)
            if sha(dst) != source[key + '_sha256']:
                raise ValueError('source copy mismatch')
            source_hashes[str(src)] = sha(src)
            source[key] = str(dst)
    rows, surfaces = [], {}
    for sex, version, target in [('male', '0-7-6', 8.3), ('female', '0-3-9', 8.0)]:
        old_id = f'{sex}-base-v{version}'
        new_id = f'{sex}-default{83 if sex == "male" else 80}-v{version}-p1'
        if new_id in chars['characters'] or any(a['characterId'] == new_id for a in catalog['assets']):
            raise ValueError('candidate ID already exists')
        old = registry['characters'][old_id]
        source = recipes['sources'][old['rules']]
        c = load_controller(source['blend'], source['controller'], 'proportions_' + sex)
        before = invariants(c)
        m, arm = c.objects()
        for label, cid, values in [
            ('latest_base', old_id, old['defaults']),
            ('candidate', new_id, {**old['defaults'], 'heads': target}),
        ]:
            c.apply_pose('T')
            c.set_controls(values, correct=False)
            values = c.controls()
            q = c.compute(values)[0]
            met = measure(c, values, q)
            if abs(met['errorHeads']) > .001 or abs(met['soleZMm']) > .03:
                raise ValueError('proportion measurement failed')
            if invariants(c) != before:
                actual = invariants(c)
                write(out / 'invariant-failure.json', {k: {'before': before[k], 'after': actual[k]} for k in before if before[k] != actual[k]})
                raise ValueError('source invariants changed: ' + ','.join(k for k in before if before[k] != actual[k]))
            bundle = c.export_fixed(str(out / 'roundtrips'))
            export = json.loads((bundle / 'manifest.json').read_text())
            row = {'sex': sex, 'label': label, 'characterId': cid, 'source': source,
                   'controls': values, 'measurement': met, 'invariants': before,
                   'fingerCalibration': c.hl_calibrate(arm),
                   'neutralExport': str(bundle), 'fbxSha256': export['fbx_sha256'],
                   'roundtrip': {k: export[k] for k in ('roundtrip_max_error_mm', 'roundtrip_joint_error_mm', 'roundtrip_weight_error')},
                   'geometry': c.validate_geometry(values)}
            rows.append(row)
            surfaces[sex + '_' + label] = q.astype('<f4')
            surfaces[sex + '_faces'] = c.source()['faces']
            print('PROPORTION', sex, label, json.dumps(met), flush=True)
            if label == 'candidate':
                fbx = out / 'fbx' / (cid + '.fbx')
                fbx.parent.mkdir(exist_ok=True)
                shutil.copy2(bundle / 'model.fbx', fbx)
                recipe = copy.deepcopy(old)
                recipe['defaults'] = values
                recipe['asset_sha256'] = sha(fbx)
                # Preset is an authored evaluator anchor, not the product card
                # identity. Explicit defaults override its heads control.
                recipe['preset'] = old['preset']
                for control in recipe['controls']:
                    if control['key'] == 'heads':
                        control['default'] = target
                recipes['characters'][cid] = recipe
                a = copy.deepcopy(next(a for a in catalog['assets'] if a['characterId'] == old_id))
                a.update(bodyId=cid, characterId=cid, bodyVersion=version.replace('-', '.') + '.p1',
                         assetSha256=sha(fbx), displayName=('남성 · 8.3등신' if sex == 'male' else '여성 · 8등신'),
                         gender=sex, role='default', order=0 if sex == 'male' else 1)
                # An old runtime does not change neutral rest controls simply
                # because catalog defaults changed. Rebuild it before activation.
                a.pop('shapeRuntime', None)
                a.pop('neutralPreview', None)
                for control in a['shape']['controls']:
                    if control['key'] == 'heads':
                        control['default'] = target
                catalog['assets'].append(a)
                chars['characters'][cid] = {**chars['characters'][old_id],
                    'display_name': a['displayName'], 'sha256': sha(fbx),
                    'artifact_uri_env': 'BODY_' + cid.upper().replace('-', '_') + '_URI',
                    'revision': a['bodyVersion']}
                row['defaultShapeSha256'] = digest_shape(cid, sha(fbx), {x['key']: x['default'] for x in recipe['controls']})
                row['catalogAsset'] = a
                row['changedControls'] = {k: [old['defaults'][k], v] for k, v in values.items() if v != old['defaults'][k]}
        c.unregister()
    # Catalog presentation changes are separate from immutable BodyRef identity.
    for a in catalog['assets']:
        if a['characterId'] in ('male-base-v0-7-2', 'female-base-v0-3-4', 'body-male-base', 'body-female-base'):
            a.update(gender='female' if 'female' in a['characterId'] else 'male', role='previous_default', order=10)
            a['displayName'] += ' · 기존 기본형'
    catalog['version'] += '-default-proportions-review-p1'
    catalog['defaultCharacterId'] = 'male-default83-v0-7-6-p1'
    write(out / 'body-models.review.json', catalog)
    write(out / 'characters.local.json', chars)
    write(out / 'shape-bakes.local.json', recipes)
    write(out / 'validation.json', {'status': 'neutral_validated_pose_checks_pending', 'rows': rows,
          'sourceHashes': source_hashes, 'activationAllowed': False,
          'existingCatalogAssetCount': len(base_catalog['assets']),
          'existingRecipesPreserved': all(recipes['characters'][k] == v for k, v in registry['characters'].items()),
          'existingRegistryPreserved': all(chars['characters'][k] == v for k, v in json.loads((args.catalog_package / 'characters.json').read_text())['characters'].items())})
    np.savez_compressed(out / 'review-surfaces.npz', **surfaces)
    load_registry(out / 'shape-bakes.local.json')
    if any(sha(p) != h for p, h in source_hashes.items()):
        raise ValueError('original source changed during build')


if __name__ == '__main__':
    main()
