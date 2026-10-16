"""Build immutable shape-bake inputs from approved body sources. Run with Blender (or pip bpy).

    blender --background --python scripts/build_shape_release.py -- \
        --source /path/male/model --source /path/female/model --out artifacts/shape-release-<date>

Each ``--source`` directory holds exactly one ``*.blend`` and its ``standin_body.py``.
Nothing about a body version is hard-coded here: the rules name is the controller's
own ``VERSION``, presets and their names come from the controller's ``PRESETS``, and
each preset becomes one catalog asset (neutral T-pose FBX) with its own slider spec.

Outputs (all local absolute paths; ``package_shape_staging.py`` rewrites them for deployment):
  sources/<rules>.blend|.py           byte copies of the approved sources
  fbx/<characterId>.fbx               neutral preset FBX (catalog asset; asset_sha256)
  shape-bakes.local.json              converter registry (shape-bake.v2)
  body-models.local.json              BFF manifest assets (shape, shapeSource; runtime added later)
  characters.local.json               converter character registry
  source-hashes.json                  input file → SHA-256
"""
import argparse, hashlib, importlib.util, json, math, re, shutil, sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.shape_spec import REGISTRY_SCHEMA, load_registry, on_grid  # noqa: E402

# Slider order: the first `basic` keys are shown by default, the rest under details.
BASIC = {
    'male': ['height_cm', 'heads', 'physique', 'muscle', 'muscle_bulk', 'shoulders'],
    'female': ['height_cm', 'heads', 'shoulders', 'muscle', 'physique', 'fat_gain', 'breast_size'],
}
LABELS = {'height_cm': '키', 'heads': '등신', 'shoulders': '어깨너비', 'muscle': '근육량', 'physique': '지방',
          'fat_gain': '지방 볼륨', 'muscle_bulk': '근육 볼륨', 'breast_size': '가슴 크기',
          'breast_projection': '가슴 돌출도', 'breast_width': '가슴 너비', 'breast_height': '가슴 높이',
          'breast_spacing': '가슴 간격', 'fat_distribution': '지방 분포', 'chest_width': '흉곽 너비',
          'chest_depth': '흉곽 두께', 'head_scale': '머리 크기'}
VERSION_RE = re.compile(r"^VERSION\s*=\s*['\"]([^'\"]+)['\"]\s*$", re.M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def step_for(key):
    return 1.0 if key == 'height_cm' else 1e-6 if key == 'heads' else 0.01


def grid_control(key, lo, hi, default, label):
    """Anchor the slider grid so the preset default is itself a reachable value.

    The BFF, the converter and HTML range inputs all validate ``(v - min) / step``;
    a default off that grid (e.g. head_scale .927 with min .90, step .01) would make
    every request for the preset fail. The minimum moves up by less than one step.
    """
    step = step_for(key)
    default = round(float(default), 6)
    n = math.floor((default - lo) / step + 1e-6)
    lo_aligned = round(default - n * step, 6)
    control = {'key': 'height' if key == 'height_cm' else key, 'min': lo_aligned, 'max': float(hi),
               'step': step, 'default': default}
    assert lo <= lo_aligned <= default <= hi and on_grid(default, control), (key, control)
    if label:
        control['label'] = label[:24]
    return control


def label_for(key, controller_label):
    if key in LABELS:
        return LABELS[key]
    return re.split(r' · | \(', controller_label)[0].strip() or None


def load_controller(blend, controller, name):
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    spec = importlib.util.spec_from_file_location(name, str(controller))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register()
    return module


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', action='append', required=True, help='directory with one .blend and standin_body.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--catalog-version', default=None)
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty; releases are immutable')
    out.mkdir(parents=True, exist_ok=True)
    sources, recipes, bodies, characters, hashes = {}, {}, [], {}, {}
    for directory in map(Path, args.source):
        blends = sorted(directory.glob('*.blend'))
        controller = directory / 'standin_body.py'
        if len(blends) != 1 or not controller.is_file():
            raise SystemExit(f'{directory}: need exactly one .blend and standin_body.py')
        declared = VERSION_RE.findall(controller.read_text(encoding='utf-8'))
        if len(declared) != 1:
            raise SystemExit(f'{controller}: one VERSION literal required')
        rules = declared[0]
        if rules in sources:
            raise SystemExit(f'duplicate source {rules}')
        sex = 'female' if '-female-' in rules else 'male'
        ver = rules.rsplit('-', 1)[1]
        copies = {}
        for key, src, ext in (('blend', blends[0], '.blend'), ('controller', controller, '.py')):
            dest = out / 'sources' / (rules + ext)
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(src, dest)
            digest = sha(src)
            assert sha(dest) == digest
            copies[key] = str(dest)
            copies[key + '_sha256'] = digest
            hashes[str(src)] = digest
        sources[rules] = copies
        sb = load_controller(copies['blend'], copies['controller'], 'release_' + sex)
        if sb.VERSION != rules:
            raise SystemExit(f'{controller}: module VERSION {sb.VERSION} != {rules}')
        mesh, _ = sb.objects()
        order = [k for k in BASIC[sex] if k in sb.FIELDS] + [k for k in sb.FIELDS if k not in BASIC[sex]]
        basic = len([k for k in BASIC[sex] if k in sb.FIELDS])
        for preset, (preset_label, _overrides) in sb.PRESETS.items():
            values = sb.normalize(sb.preset_values(preset, mesh))
            sb.set_controls(values, correct=False)
            values = sb.controls()
            exported = sb.export_fixed(str(out / '.neutral-exports'))
            meta = json.loads((exported / 'manifest.json').read_text())
            cid = f"{sex}-{preset.lower().replace('_', '-')}-v{ver.replace('.', '-')}"
            fbx = out / 'fbx' / (cid + '.fbx')
            fbx.parent.mkdir(exist_ok=True)
            shutil.copy2(exported / 'model.fbx', fbx)
            assert sha(fbx) == meta['fbx_sha256']
            controls = [grid_control(k, *sb.FIELDS[k][:2], values[k], label_for(k, sb.FIELDS[k][3])) for k in order]
            recipes[cid] = {'rules': rules, 'preset': preset, 'asset': mesh['asset_identity'],
                            'asset_sha256': meta['fbx_sha256'], 'defaults': values,
                            'controls': [{k: v for k, v in c.items() if k != 'label'} for c in controls]}
            bodies.append({
                'bodyId': cid, 'bodyVersion': ver, 'characterId': cid, 'assetSha256': meta['fbx_sha256'],
                'rigVersion': sb.RIG_VERSION, 'measurementVersion': 'shape-authoring-v1',
                'displayName': ('남성' if sex == 'male' else '여성') + ' · ' + preset_label,
                'supportedPoseIds': [],
                'shape': {'schemaVersion': 'body-shape.v1', 'controls': controls, 'basic': basic},
                'shapeSource': {'rules': rules, 'blendSha256': copies['blend_sha256'],
                                'controllerSha256': copies['controller_sha256']},
            })
            characters[cid] = {'display_name': preset_label, 'artifact_uri_env': 'BODY_' + cid.upper().replace('-', '_') + '_URI',
                               'sha256': meta['fbx_sha256'], 'rig_profile': 'mixamo', 'revision': ver}
            print('SHAPE_ASSET_READY', cid, meta['fbx_sha256'][:12], flush=True)
    shutil.rmtree(out / '.neutral-exports', ignore_errors=True)
    registry = {'schema': REGISTRY_SCHEMA, 'sources': sources, 'characters': recipes}
    (out / 'shape-bakes.local.json').write_text(json.dumps(registry, ensure_ascii=False, indent=2))
    load_registry(out / 'shape-bakes.local.json')  # same checks the converter runs at start-up
    version = args.catalog_version or 'shape-' + '-'.join(sorted(sources))
    (out / 'body-models.local.json').write_text(json.dumps(
        {'version': version, 'defaultCharacterId': bodies[0]['characterId'], 'assets': bodies}, ensure_ascii=False, indent=2))
    (out / 'characters.local.json').write_text(json.dumps({'schema_version': 1, 'characters': characters}, ensure_ascii=False, indent=2))
    (out / 'source-hashes.json').write_text(json.dumps(hashes, indent=2))
    print('SHAPE_RELEASE_READY', out, len(recipes), 'assets', sorted(sources), flush=True)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
