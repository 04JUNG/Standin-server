"""Port the pinned breast7 residuals onto the active 0.4.0 source, never re-sculpt.

Run in Blender with --disable-autoexec. All inputs are immutable and byte pinned.
Only three named BODY keys and the breast composition policy are owned here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from build import controller_text as donor_controller_text
from verify import invariants
sys.path.insert(0, str(Path(__file__).parents[1] / 'character_upgrade'))
from build_source import projection

VERSION = 'standin-female-body-0.4.0.breast7.2'
PINS = {
    'blend': '5a62d2f3fb49d77fdc04d1c78a361edf9fbab55e71e0e47d3b6b393fae14d3a9',
    'controller': '1164dff9a4c6069d0da37b80dd26c61544bc6f9eb9c9f3469c33d457e893e9af',
    'donor_blend': 'c2d3a81e52eda54c0fed3d23aa3e45375f99989796f10fab9f2578e0d98a843a',
    'donor_controller': '1cdb04c45e34ecf054da0769479f9f28c5e62a8c6f5154340b1b28a25789ae2d',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(blend, path, name):
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    spec = importlib.util.spec_from_file_location(name, path)
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    sb.register()
    return sb


def controller_text(text, asset, presets, pose_guard=False):
    # Reuse the actual merged breast policy, not a reimplementation or rule alias.
    entry = "if __name__=='__main__':register()"
    if text.count(entry) != 1:
        raise ValueError('Pinned current controller entry point changed')
    text = text.replace(entry, '')
    for old, new in {
        "VERSION='standin-female-body-0.4.0'": f"VERSION='{VERSION}'",
        "'breast_size':(-1.,1.,0.,'가슴 크기')": "'breast_size':(-1.,2.,0.,'가슴 크기')",
        "out['BODY_BreastLarge']=max(0.,c['breast_size'])":
        "out['BODY_BreastLarge']=min(1.,max(0.,c['breast_size']))\n"
        "    out['BODY_BreastExtended']=max(0.,c['breast_size']-1.)",
    }.items():
        if text.count(old) != 1:
            raise ValueError('Pinned current controller structure changed: ' + old)
        text = text.replace(old, new)
    # controller_text() validates the original donor structure before emitting its
    # suffix. Obtain that suffix from the pinned original parent syntax.
    template = ("VERSION='standin-female-body-0.3.4'\n'version':(0,3,4)\n"
                "'breast_size':(-1.,1.,0.,'가슴 크기')\n"
                "out['BODY_BreastLarge']=max(0.,c['breast_size'])\n" + entry)
    suffix = donor_controller_text(template, asset).split("\nBREAST_RULES=", 1)[1]
    suffix = ('\nBREAST_RULES=' + suffix).replace(
        "BREAST_PARENT_RULES='standin-female-body-0.3.4'",
        "BREAST_PARENT_RULES='standin-female-body-0.4.0'").replace(
        'pinned female 0.3.4 source', 'pinned female 0.4.0 source')
    # 0.4.0 uses approved PSDs, independent of breast shape. Its correct_pose is
    # retained as the reference implementation by the exact breast policy above.
    if pose_guard:
        suffix = suffix.replace("        qa=json.loads(mesh.get('surface_qa','{}'))", "        qa=json.loads(mesh.get('surface_qa','{}'))\n        reference_surface=evaluated(mesh)")
        suffix = suffix.replace("        LAST_SIGNATURE=pose_signature(scene)", "        breast_pose_contact_guard(scene, reference_surface)\n        LAST_SIGNATURE=pose_signature(scene)")
        suffix = suffix.replace(entry, Path(__file__).with_name('pose_guard040.py').read_text() + '\n' + entry)
    return text + '\nPRESETS=' + repr(presets) + '\n' + suffix


def main(args):
    global VERSION
    if args.pose_contact_guard:
        VERSION = "standin-female-body-0.4.0.breast7.4"
    for name, expected in PINS.items():
        if sha(getattr(args, name)) != expected:
            raise ValueError('Input bytes differ from pinned source: ' + name)
    if args.out.exists():
        raise ValueError('Use a new immutable output directory')
    donor = load(args.donor_blend, args.donor_controller, 'donor')
    ds = donor.source()
    donor_co, donor_faces = ds['co'].copy(), ds['faces'].copy()
    residuals = {
        'BODY_BreastSmall': ds['deltas']['BODY_BreastSmall'] - ds['deltas']['BODY_BreastSmallReference'],
        'BODY_BreastExtended': ds['deltas']['BODY_BreastExtended'].copy(),
    }
    donor.unregister()
    parent = load(args.blend, args.controller, 'current')
    m, a = parent.objects()
    s = parent.source()
    before = invariants(parent)
    original = {k: v.copy() for k, v in s['deltas'].items()}
    keys_before = {k.name: parent.arr(k.data).copy() for k in m.data.shape_keys.key_blocks}
    parent_asset = m['asset_identity']
    ids, bary, distance = projection(s['co'], donor_co, donor_faces)
    tree = KDTree(len(donor_co))
    for i, p in enumerate(donor_co):
        tree.insert(p, i)
    tree.balance()
    nearest = [tree.find(p) for p in s['co']]
    ni = np.array([x[1] for x in nearest])
    nd = np.array([x[2] for x in nearest])
    exact = nd == 0
    transfers, transfer_report = {}, {}
    for name, delta in residuals.items():
        transferred = np.einsum('ni,nij->nj', bary, delta[ids])
        transferred[exact] = delta[ni[exact]]
        significant = np.linalg.norm(delta, axis=1) > 1e-6
        missing = np.setdiff1d(np.flatnonzero(significant), ni[exact])
        if len(missing):
            raise ValueError('Authored breast vertices do not match: ' + str(missing))
        moved = np.linalg.norm(transferred, axis=1) > 1e-6
        if np.any((distance > 1e-7) & moved):
            raise ValueError('Meaningful displacement would require approximate transfer')
        transfers[name] = transferred
        transfer_report[name] = {
            'significant_donor_vertices': int(significant.sum()),
            'significant_missing_vertices': missing.tolist(),
            'max_nonexact_displacement_mm': float(np.linalg.norm(transferred[~exact], axis=1).max(initial=0) * 1000),
            'exact_significant_transfer': True,
        }
    # Preserve the current negative target as the reference and add the donor
    # Small-minus-legacy residual; Extended is already an additive +1..+2 field.
    changed = {
        'BODY_BreastSmall': original['BODY_BreastSmall'] + transfers['BODY_BreastSmall'],
        'BODY_BreastSmallReference': original['BODY_BreastSmall'],
        'BODY_BreastExtended': transfers['BODY_BreastExtended'],
    }
    parent.BUSY = True
    for name, delta in changed.items():
        block = m.data.shape_keys.key_blocks.get(name) or m.shape_key_add(name=name, from_mix=False)
        block.data.foreach_set('co', (s['co'] + delta).ravel())
        block.value = 0
    for name, co in keys_before.items():
        if name != 'BODY_BreastSmall' and not np.array_equal(co, parent.arr(m.data.shape_keys.key_blocks[name].data)):
            raise ValueError('Unowned shape key changed: ' + name)
    all_ds = {k.name: parent.arr(k.data) - s['co'] for k in m.data.shape_keys.key_blocks if k.name.startswith('BODY_')}
    body_sha = hashlib.sha256(b''.join(n.encode() + all_ds[n].astype('<f4').tobytes() for n in sorted(all_ds))).hexdigest()
    revision = VERSION.rsplit('.', 1)[-1]
    asset = 'standin-female040-breast7-' + revision + '-' + body_sha[:20]
    registry = json.loads(args.registry.read_text())
    presets = {r['preset']: (r['preset'], r['defaults']) for r in registry['characters'].values() if r['rules'] == 'standin-female-body-0.4.0'}
    if len(presets) != 11:
        raise ValueError('Expected all eleven current female presets')
    out = args.out.resolve()
    model = out / 'model'
    model.mkdir(parents=True)
    controller = model / 'standin_body.py'
    controller.write_text(controller_text(args.controller.read_text(), parent_asset, presets, args.pose_contact_guard))
    lock = json.loads(m['asset_lock'])
    lock.update(rules=VERSION, body_deltas_sha256=body_sha, parent_asset=parent_asset,
                parent_blend_sha256=PINS['blend'], parent_controller_sha256=PINS['controller'],
                donor_blend_sha256=PINS['donor_blend'], donor_controller_sha256=PINS['donor_controller'],
                controller_sha256=sha(controller), builder_sha256=sha(__file__))
    m['asset_identity'] = asset
    m['asset_lock'] = parent.canonical(lock)
    m['breast_candidate'] = parent.canonical({'rules': VERSION, 'range': [-1, 2], 'donor_rules': donor.VERSION})
    parent.unregister()
    spec = importlib.util.spec_from_file_location('ported', controller)
    child = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(child)
    child.register()
    child.source()
    after = invariants(child)
    if before != after:
        raise ValueError('Current model invariants changed')
    for t in list(bpy.data.texts):
        bpy.data.texts.remove(t)
    bpy.context.preferences.filepaths.save_version = 0
    blend = model / (VERSION + '.blend')
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True)
    report = {'status': 'unvalidated_candidate', 'rules': VERSION, 'rig_rules': child.RIG_VERSION,
              'asset': asset, 'parent': {'rules': parent.VERSION, 'asset': parent_asset,
              'blend_sha256': PINS['blend'], 'controller_sha256': PINS['controller']},
              'source': {'blend': str(blend), 'blend_sha256': sha(blend),
                         'controller': str(controller), 'controller_sha256': sha(controller)},
              'donor': {'rules': donor.VERSION, 'blend_sha256': PINS['donor_blend'], 'controller_sha256': PINS['donor_controller']},
              'transfer': transfer_report, 'invariants_before': before, 'invariants_after': after,
              'all_unowned_shape_keys_exact': True, 'changed_keys': list(changed),
              'breast_size': {'min': -1, 'max': 2, 'step': .01, 'default': 0},
              'preset_defaults': presets, 'lock': lock}
    (out / 'candidate.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    np.savez_compressed(out / 'transfer.npz', ids=ids, bary=bary, distance=distance, nearest_index=ni, nearest_distance=nd,
                        **{'donor_' + k: v for k, v in residuals.items()}, **transfers)
    print('PORT_READY', out, asset, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in PINS:
        ap.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    ap.add_argument('--pose-contact-guard', action='store_true')
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    main(ap.parse_args(sys.argv[sys.argv.index('--') + 1:]))
