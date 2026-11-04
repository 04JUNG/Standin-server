"""Bind validated local assets, preserving original files and old identities."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.shape_spec import digest_shape


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


ap = argparse.ArgumentParser(__doc__)
ap.add_argument('--release', type=Path, required=True)
ap.add_argument('--package', type=Path, required=True)
ap.add_argument('--old-package', type=Path, required=True)
ap.add_argument('--confirmation', type=Path, required=True)
ap.add_argument('--manifest', type=Path, required=True)
args = ap.parse_args()
r = args.release.resolve()
validation = read(r / 'validation.json')
poses = read(r / 'pose-qa/results.json')
inspection = read(r / 'bake-qa/independent-inspection.json')
runtime_qa = read(r / 'runtime-qa/results.json')
catalog = read(r / 'body-models.review.json')
published = read(args.package / 'body-models.json')
characters = read(r / 'characters.local.json')
registry = read(r / 'shape-bakes.local.json')
old_registry = read(args.package / 'shape-bakes.json')
if poses['status'] != 'complete' or len(poses['rows']) != 24 or len(inspection) != 4 or not all(x['passed'] for x in inspection):
    raise ValueError('incomplete pose or independent inspection')
if any(x['increase'] > 0 for x in poses['pairComparisons']):
    raise ValueError('new intersection pairs require review')
if not all(registry['characters'][k] == v for k, v in old_registry['characters'].items()):
    raise ValueError('existing recipe changed')
if not all(characters['characters'][k] == v for k, v in read(args.package / 'characters.json')['characters'].items()):
    raise ValueError('existing character changed')
preserved = r / 'preserved'
preserved.mkdir(exist_ok=True)
records = []


def preserve(src, relative, expected=None):
    src = Path(src)
    digest = sha(src)
    if expected is not None and digest != expected:
        raise ValueError('original hash mismatch: ' + str(src))
    dst = preserved / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha(dst) != digest:
        raise ValueError('preserved immutable copy already differs')
    if not dst.exists():
        shutil.copy2(src, dst)
    if sha(dst) != digest:
        raise ValueError('preservation copy failed')
    records.append({'source': str(src), 'copy': str(dst), 'sha256': digest, 'bytes': src.stat().st_size})


for name in ('body-models.json', 'characters.json', 'shape-bakes.json', 'manifest-bff.json', 'manifest-converter.json'):
    preserve(args.package / name, Path('published') / name)
for name in ('body-models.json', 'characters.json', 'shape-bakes.json', 'manifest.json'):
    preserve(args.old_package / name, Path('production') / name)
preserve(args.confirmation, Path('CONFIRMED_MODELS.json'))
previous = read(r / 'previous-defaults.json')
for model in previous:
    recipe = old_registry['characters'][model['characterId']]
    model['shapeSha256'] = digest_shape(model['characterId'], model['fbxSha256'],
                                      {c['key']: c['default'] for c in recipe['controls']})
for cid in ('body-male-base', 'body-female-base', 'male-base-v0-7-2', 'female-base-v0-3-4', 'male-base-v0-7-6', 'female-base-v0-3-9'):
    asset = next(a for a in published['assets'] if a['characterId'] == cid)
    filename = {'body-male-base': 'male-base', 'body-female-base': 'female-base'}.get(cid, cid) + '.fbx'
    src = args.package / 'fbx' / filename
    if not src.exists():
        src = args.old_package / 'fbx' / filename
    preserve(src, Path('fbx') / filename, asset['assetSha256'])
for original, digest in validation['sourceHashes'].items():
    if sha(original) != digest:
        raise ValueError('original authoring source changed')
for source in registry['sources'].values():
    for key in ('blend', 'controller'):
        if sha(source[key]) != source[key + '_sha256']:
            raise ValueError('copied authoring source changed')
for a in catalog['assets']:
    if a.get('role') != 'default':
        continue
    cid = a['characterId']
    sex = a['gender']
    model = next(x for x in validation['rows'] if x['characterId'] == cid)
    if set(model['changedControls']) != {'heads'} or sha(r / 'fbx' / (cid + '.fbx')) != a['assetSha256']:
        raise ValueError('candidate changed more than heads or FBX mismatch')
    if not all(x['sourceMatches'] and x['controlsMatch'] for x in runtime_qa['rows'] if x['sex'] == sex):
        raise ValueError('runtime source or controls mismatch')
    old = next(x for x in published['assets'] if x['characterId'] == ('male-base-v0-7-6' if sex == 'male' else 'female-base-v0-3-9'))
    runtime = copy.deepcopy(old['shapeRuntime'])
    src = args.package / 'runtime' / Path(runtime['path']).name
    if sha(src) != runtime['sha256'] or src.stat().st_size != runtime['bytes']:
        raise ValueError('runtime bytes mismatch')
    target = r / 'runtime' / src.name
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(src, target)
    runtime['path'] = str(target)
    runtime['preset'] = 'BASE'
    a['shapeRuntime'] = runtime
    neutral = next(x for x in read(r / 'catalog-render/body-models.local.json')['assets'] if x['characterId'] == cid)['neutralPreview']
    if sha(neutral['path']) != neutral['sha256'] or neutral['characterSha256'] != a['assetSha256']:
        raise ValueError('neutral preview lineage mismatch')
    a['neutralPreview'] = neutral
write(r / 'body-models.review.json', catalog)
write(r / 'preservation.json', {'originalsUnchanged': True, 'records': records,
      'allExistingCharacterRecipesUnchanged': True, 'allExistingCharacterRegistryEntriesUnchanged': True,
      'sourceCopiesByteIdentical': True, 'originalSourceHashes': validation['sourceHashes']})
manifest = read(args.manifest)
manifest['status'] = 'model_validation_complete_visual_approval_pending'
manifest['activationAllowed'] = False
manifest['previousDefaults'] = previous
manifest['preservationPath'] = str(r / 'preservation.json')
manifest['validation'] = {'neutralRoundtrips': 4, 'poseComparisons': 12, 'poseRoundtrips': 12,
    'actualConverterRuns': 8, 'independentInspectionPairs': 4,
    'maximumFinalSurfaceErrorMm': max(x['shape_effect']['max_surface_error_mm'] for x in inspection),
    'maximumFinalJointErrorMm': max(x['joints']['max_error_mm'] for x in inspection),
    'minimumDeltaCorrelation': min(x['shape_effect']['magnitude_correlation'] for x in inspection),
    'newIntersectionPairs': 0, 'sampledRuntime': runtime_qa,
    'samePoseInspectionAdapter': 'verified shoulder signature replay, reviewed precedent PR155; original .01mm/correlation gates unchanged'}
manifest['catalogContract']['newRuntimeStatus'] = 'reused exact SSR2 bytes after source/controls and actual client TS surface comparison; BASE is evaluator anchor, card defaults override heads'
manifest['catalogContract']['newNeutralPreviewStatus'] = 'rendered from exact source using existing body-comparison.v1 protocol; FBX SHA pinned'
manifest['review'] = {'url': 'http://127.0.0.1:8833/', 'directory': str(r / 'review'),
    'renderProtocol': str(r / 'render-protocol.json'), 'visualApproval': 'pending_user',
    'screenshots': [str(p) for p in sorted((r / 'review').glob('browser-*.jpg'))],
    'comparisonSheets': [str(r / 'review' / (sex + '-three-view-sheet.png')) for sex in ('male', 'female')]}
for sex, m in manifest['models'].items():
    asset = next(a for a in catalog['assets'] if a.get('gender') == sex and a.get('role') == 'default')
    m.update(catalogAsset=asset, validation='model_checks_passed; visual_approval_pending',
             shapeRuntime=asset['shapeRuntime'], neutralPreview=asset['neutralPreview'])
    m['previousCharacterId'] = m['currentEditorPreviousCharacterId']
    baseline = registry['characters'][m['latestSourceBaseCharacterId']]
    m['latestSourceBaseShapeSha256'] = digest_shape(m['latestSourceBaseCharacterId'], baseline['asset_sha256'],
                                                   {c['key']: c['default'] for c in baseline['controls']})
manifest['remaining'] = ['User visual approval and integrated application UI approval',
    'Prepare reviewed deployment package with immutable existing assets plus new IDs; no deployment authorized',
    'Rebuild any canonical pose posters/templates/GLB caches only under new character/shape identities; preserve old cache entries',
    'CSP native import/locks and breast7 combination are separate work; neither changed or claimed here']
write(args.manifest, manifest)
print('FINALIZED_REVIEW_ONLY', args.manifest)
