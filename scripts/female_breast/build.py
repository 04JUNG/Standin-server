"""Build an isolated, immutable breast candidate from confirmed female 0.3.4.

blender -b --disable-autoexec --python scripts/female_breast/build.py -- \
  --blend /official/standin-female-body-0.3.4.blend \
  --controller /official/standin-female-body-0.3.4.py --out /new/candidate
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sculpt import targets

VERSION = 'standin-female-body-0.3.4.breast6'
PARENT_BLEND = 'a3622a9d4ae7fa2e88aa3903db102a3f9c4660b3fb34ce376950e8cd2f2bae32'
PARENT_CONTROLLER = 'df981aedc3784d38e008f6ddb7c7108ce26d8a528ef31539facdabfe2937eb1f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register()
    return module


def controller_text(text, asset):
    replacements = {
        "VERSION='standin-female-body-0.3.4'": f"VERSION='{VERSION}'",
        "'version':(0,3,4)": "'version':(0,3,4,6)",
        "'breast_size':(-1.,1.,0.,'가슴 크기')": "'breast_size':(-1.,2.,0.,'가슴 크기')",
        "out['BODY_BreastLarge']=max(0.,c['breast_size'])":
        "out['BODY_BreastLarge']=min(1.,max(0.,c['breast_size']))\n"
        "    out['BODY_BreastExtended']=max(0.,c['breast_size']-1.)",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError('Pinned controller structure changed: '+old)
        text = text.replace(old, new)
    # The Blender Text Editor entry point must register after the new wrappers,
    # just like the converter's import-then-register path.
    entry = "if __name__=='__main__':register()"
    if text.count(entry) != 1:
        raise ValueError('Pinned controller entry point changed')
    text = text.replace(entry, '')
    # Exact source identity is required; a stored preset is never re-labelled in place.
    text += f'''

BREAST_RULES='local-mound-rounded-upper-and-lower-pole-2'
BREAST_PARENT_RULES='standin-female-body-0.3.4'
BREAST_PARENT_ASSET='{asset}'
BREAST_REFERENCE=False
BREAST_PARENT_MAPPINGS=mappings
BREAST_PARENT_COMPUTE=compute

def mappings(c):
    out=BREAST_PARENT_MAPPINGS(c)
    out['BODY_BreastSmallReference']=0.
    if BREAST_REFERENCE:
        out['BODY_BreastSmallReference']=out['BODY_BreastSmall']
        out['BODY_BreastSmall']=0.
        out['BODY_BreastExtended']=0.
    return out

def breast_delta(s,c):
    small=max(0.,-c['breast_size']);large=max(0.,c['breast_size']-1.)
    return ((s['deltas']['BODY_BreastSmall']-s['deltas']['BODY_BreastSmallReference'])*small
            +s['deltas']['BODY_BreastExtended']*large)

def compute(c,scene=None):
    """Compose breast changes locally after the inherited fat/muscle pipeline."""
    global BREAST_REFERENCE
    if BREAST_REFERENCE or 0<=c['breast_size']<=1:
        return BREAST_PARENT_COMPUTE(c,scene)
    previous=BREAST_REFERENCE
    try:
        BREAST_REFERENCE=True
        reference_body,_,reference=BREAST_PARENT_COMPUTE(c,scene)
    finally:BREAST_REFERENCE=previous
    s=source(scene);mesh,_=objects(scene);delta=breast_delta(s,c)
    offset=warp(s['co']+delta,c,mesh)-warp(s['co'],c,mesh)
    out=fold_guard(s,reference_body,reference_body+offset,max_iterations=20,max_rings=3)
    return out,mappings(c),reference

BREAST_PARENT_CORRECT_POSE=correct_pose

def correct_pose(scene=None,check_intersections=False,contacts=False):
    """Change only the breast morph; retain the inherited pose residual.

    The surface selector scores whole limb regions. Re-running it on a larger
    breast can switch the selected arm deformation and move the back axilla.
    The extension uses +1; reduction uses the legacy small reference residual.
    No new displacement is authored on the arm, rib cage, back or hand.
    """
    global BUSY,LAST_SIGNATURE,BREAST_REFERENCE
    scene=scene or bpy.context.scene
    values=controls(scene)
    if BREAST_REFERENCE or 0<=values['breast_size']<=1:
        return BREAST_PARENT_CORRECT_POSE(scene,check_intersections,contacts)
    old_busy=BUSY;old_reference=BREAST_REFERENCE
    try:
        BUSY=False
        BREAST_REFERENCE=True
        set_controls(values,scene,correct=False)
        BREAST_PARENT_CORRECT_POSE(scene,check_intersections,contacts)
        mesh,_=objects(scene)
        block=mesh.data.shape_keys.key_blocks.get('AUTO_Surface')
        saved=(arr(block.data).copy(),float(block.value)) if block is not None else None
        qa=json.loads(mesh.get('surface_qa','{{}}'))
        BREAST_REFERENCE=old_reference
        set_controls(values,scene,correct=False)
        if saved is not None:
            block.data.foreach_set('co',saved[0].ravel());block.value=saved[1]
        qa.update(breast_pose_reference_size=min(1.,values['breast_size']),breast_changes_local_only=True)
        mesh['surface_qa']=canonical(qa);mesh['pose_qa']=canonical(qa)
        mesh.data.update();bpy.context.view_layer.update()
        LAST_SIGNATURE=pose_signature(scene)
    finally:
        BUSY=old_busy;BREAST_REFERENCE=old_reference

def migrate_breast_snapshot(data,scene=None):
    """Explicit opt-in migration; negative sizes intentionally acquire new geometry."""
    if not isinstance(data,dict) or set(data)!={{'schema','asset','rules','rig_rules','controls'}}:
        raise ValueError('Invalid source snapshot')
    if (data['schema']!=1 or data['asset']!=BREAST_PARENT_ASSET
        or data['rules']!=BREAST_PARENT_RULES or data['rig_rules']!=RIG_VERSION):
        raise ValueError('Snapshot is not the pinned female 0.3.4 source')
    values=normalize(data['controls'])
    if not -1<=values['breast_size']<=1:raise ValueError('Invalid legacy breast size')
    mesh,_=objects(scene)
    return {{'schema':1,'asset':mesh['asset_identity'],'rules':VERSION,
            'rig_rules':RIG_VERSION,'controls':values}}

if __name__=='__main__':register()
'''
    return text


def main(args):
    if sha(args.blend) != PARENT_BLEND or sha(args.controller) != PARENT_CONTROLLER:
        raise ValueError('Source bytes differ from confirmed female 0.3.4')
    out = args.out.resolve()
    if out.exists():
        raise ValueError('Candidate destination already exists; use a new directory')
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    parent = load(args.controller.resolve(), 'breast_parent')
    m, a = parent.objects()
    source = parent.source()
    lock = json.loads(m['asset_lock'])
    old_asset = m['asset_identity']
    changed = targets(source['co'], source['deltas'])
    original = {k:v.copy() for k,v in source['deltas'].items()}
    parent.BUSY = True
    for name, delta in changed.items():
        block = m.data.shape_keys.key_blocks.get(name) or m.shape_key_add(name=name, from_mix=False)
        block.data.foreach_set('co', (source['co']+delta).ravel())
        block.value = 0
    all_deltas = {k.name:parent.arr(k.data)-source['co'] for k in m.data.shape_keys.key_blocks if k.name.startswith('BODY_')}
    for name, delta in original.items():
        if name not in changed and not np.array_equal(delta, all_deltas[name]):
            raise ValueError('Unowned BODY key changed: '+name)
    body_sha = hashlib.sha256(b''.join(n.encode()+all_deltas[n].astype('<f4').tobytes() for n in sorted(all_deltas))).hexdigest()
    identity = 'standin-female-'+VERSION.rsplit('.',1)[-1]+'-'+body_sha[:20]
    lock.update(rules=VERSION, parent_blend_sha256=PARENT_BLEND,
                parent_controller_sha256=PARENT_CONTROLLER, parent_asset=old_asset,
                body_deltas_sha256=body_sha, builder_sha256=sha(__file__),
                sculpt_sha256=sha(Path(__file__).with_name('sculpt.py')))
    text = controller_text(args.controller.read_text(), old_asset)
    out.mkdir(parents=True)
    model = out/'model'
    model.mkdir()
    controller = model/'standin_body.py'
    controller.write_text(text)
    lock['controller_sha256'] = sha(controller)
    m['asset_identity'] = identity
    m['asset_lock'] = parent.canonical(lock)
    m['breast_candidate'] = parent.canonical({'status':'draft_artist_review_required',
        'parent_rules':parent.VERSION, 'rules':VERSION, 'range':[-1,2],
        'preserves':'Basis/topology/UV/rig/rest/weights/materials and all other BODY keys'})
    for t in bpy.data.texts:
        if t.name.endswith('.py'):
            t.clear()
            t.write(text)
    parent.unregister()
    child = load(controller, 'breast_candidate')
    child.set_controls(child.defaults(m), correct=False)
    # Validate the new lock with the normal controller path, without bypasses.
    child.CACHE.clear()
    child.source()
    blend = model/(VERSION+'.blend')
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True)
    manifest = {'status':'draft_artist_review_required','rules':VERSION,'rig_rules':child.RIG_VERSION,
        'asset':identity,'parent':{'rules':parent.VERSION,'asset':old_asset,
        'blend_sha256':PARENT_BLEND,'controller_sha256':PARENT_CONTROLLER},
        'source':{'blend':str(blend),'blend_sha256':sha(blend),
                  'controller':str(controller),'controller_sha256':sha(controller)},
        'lock':lock,'changed_keys':list(changed),'breast_size':{'min':-1,'max':2,'old_max':1,'default':0},
        'mapping':{'BODY_BreastSmall':'max(0,-size)','BODY_BreastLarge':'min(1,max(0,size))',
                   'BODY_BreastExtended':'max(0,size-1)'},
        'migration':'migrate_breast_snapshot explicitly; negative sizes intentionally change',
        'builder_sha256':sha(__file__),'sculpt_sha256':sha(Path(__file__).with_name('sculpt.py'))}
    (out/'candidate.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('BREAST_CANDIDATE_READY', out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--blend', type=Path, required=True)
    ap.add_argument('--controller', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    main(ap.parse_args(sys.argv[sys.argv.index('--')+1:]))
