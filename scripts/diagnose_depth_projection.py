"""Known-pose 2x2 depth diagnostic: projection x representation, no retrieval."""
import argparse
from dataclasses import asdict
import fcntl
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_z_rendered import render_pose, sha256, save_json
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_humanart_z import read_rgb
from scripts.diagnose_depth_style import fixed_infer, features, read
from scripts.confirm_perspective_isolation import strong_camera, projection_strength
from standin_eval.perspective_search import Camera, camera_points
from standin_eval.z_diagnostics import foreground_bbox, infer_pose
from standin_eval.mixed_z import corrected_rtmw, compose_query
from standin_eval.dual_label_search import LEFT_RIGHT
from standin_eval.ordinal_depth import PAIRS

STYLES = ('rough', 'solid')
PROJECTIONS = ('ortho', 'perspective')
SOURCE = ROOT/'out/z-diagnosis-20261012-heldout/answers.json'


def choose_camera(joints, index):
    # Geometry-only stimulus selection; never inspect model output or search ranks.
    first = (17., 77., 137., 227.)[index % 4]
    for offset in (0, 90, 180, 270):
        for pitch in (-27., 22.):
            camera = strong_camera(joints, (first+offset) % 360, pitch)
            if projection_strength(joints, camera) >= 1.75:
                return camera
    raise ValueError('no predeclared camera provides sufficient perspective; abort before inference')


def verify(args, seals=()):
    for name, digest in read(args.out/'protocol.json')['hashes'].items():
        assert sha256(Path(name)) == digest, name
    for seal in seals:
        for name, digest in read(args.out/seal).items():
            assert sha256(args.out/name) == digest, name


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), SOURCE, args.human_model, args.z_model, args.library_root/'poses.db']
    files += [ROOT/p for p in ('scripts/experiment_z_rendered.py', 'scripts/experiment_z_axis_search.py',
        'scripts/experiment_humanart_z.py', 'scripts/diagnose_depth_style.py',
        'scripts/confirm_perspective_isolation.py', 'standin_eval/perspective_search.py',
        'standin_eval/z_diagnostics.py', 'standin_eval/mixed_z.py', 'standin_eval/depth_correspondence.py',
        'standin_eval/dual_label_search.py', 'standin_eval/ordinal_depth.py', 'src/bvh.py',
        'standin_eval/z_axis_search.py')]
    save_json(args.out/'protocol.json', {
        'stage': 'development diagnostic, old40 families reused, not independent verification',
        'sample': 'all40 original source IDs/order; no image replacement, geometry preflight failure aborts before inference',
        'design': '2x2: ortho/perspective x rough/solid; same pose,yaw,pitch per four images;512px',
        'camera': 'yaw=(17,77,137,227)[i%4], offsets0/90/180/270 outer loop; pitches -27,22 inner loop; first body depth ratio>=1.75, distance solves ratio2 with .65torso all-joint safety; render volume safety must pass',
        'render': 'same ellipsoid geometry, seed20261014+i; style pair GT XY exactly equal; projection changes framing/foreshortening, not just a pixel warp',
        'human': 'infer only rough RGB192x256, four rotations median body score; >=6body and all shoulder/hip anchors >=.3; freeze before RTMW; share XY,mask,crop,turn with solid at same projection',
        'depth': 'corrected RTMW RGB288x384 CPU, same HumanArt crop/turn, +Zfarther, hip centered /fixed0.5m',
        'mapping': 'RTMW-to-Human wholebody LR via observedXY only, frozen map_depth; scores do not gate Z',
        'scoring': 'Human-to-GT global LR chosen by valid body XY for diagnostic scoring only; map predicted depths/masks/errors back to semanticGT slots before paired projection comparison; eight fixed limb-anchor pairs',
        'eligibility': 'fixedGT |delta|>=.1 true3Dtorso denominator; Humanmask both & predicted |delta|>=.1; secondary both XY discrepancy<=.5HumanTorso; all missing/failures abstain',
        'primary': 'within each style, ortho vs perspective common eligible semantic pairs; own denominators, fixedGT missed/abstained, constant front/back controls and image gain/loss counts; no search/adoption gate',
        'secondary': 'within each projection rough vs solid on common pairs; model agreement and score diagnostics',
        'limits': ['known40, no freshholdout/humanrough accuracy', 'same-image pairs correlated',
                   'GT-rendered solid not reconstructible from real rough by this method',
                   'camera distance and projection change framing and crop; not isolated focal parameter',
                   'human GT correspondence uses answers only during scoring; not deployable correction'],
        'hashes': {str(p.resolve()): sha256(p) for p in files}})
    sources = read(SOURCE); assert len(sources) == 40
    library = {p.pose_id:p for p in load_library(args.library_root)}
    manifest, answers = [], []
    (args.out/'images').mkdir()
    for i, source in enumerate(sources):
        pose = library[source['pose_id']]; strong = choose_camera(pose.joints, i)
        truth_z = camera_points(pose.joints, strong)[:, 2]
        record = {'sample': f'query_{i+1:03}', 'pose_id': pose.pose_id, 'family_id': pose.family_id,
                  'camera': asdict(strong), 'body_depth_ratio': projection_strength(pose.joints, strong),
                  'camera_z': truth_z.tolist(), 'projections': {}}
        for projection, cam in (('ortho', Camera(strong.yaw, strong.pitch)), ('perspective', strong)):
            item = {'sample': record['sample'], 'projection': projection, 'styles': {}}
            previous = None
            for style in STYLES:
                image, xy = render_pose(pose.joints, cam.yaw, size=512, style=style,
                                       seed=20261014+i, pitch=cam.pitch, distance=cam.distance)
                if previous is not None: assert np.array_equal(previous, xy)
                previous = xy
                name = f'{record["sample"]}_{projection}_{style}.png'
                path = args.out/'images'/name; image.save(path)
                item['styles'][style] = {'path': str(path.resolve()), 'sha256': sha256(path)}
            manifest.append(item); record['projections'][projection] = xy.tolist()
        answers.append(record)
    save_json(args.out/'manifest.json', manifest); save_json(args.out/'answers.json', answers)
    save_json(args.out/'prepared.json', {p: sha256(args.out/p) for p in ('manifest.json', 'answers.json')})
    verify(args, ('prepared.json',))
    print('prepared', len(manifest)*2, 'images; min/median/max depth ratio',
          np.quantile([r['body_depth_ratio'] for r in answers], [0,.5,1]).tolist(), flush=True)


def human(args):
    verify(args, ('prepared.json',)); assert not (args.out/'human.json').exists()
    from rtmlib import RTMPose
    model = RTMPose(str(args.human_model), model_input_size=(192,256), backend='onnxruntime', device='cpu')
    records = []; (args.out/'human-raw').mkdir(exist_ok=False)
    for item in read(args.out/'manifest.json'):
        entry = item['styles']['rough']; assert sha256(Path(entry['path'])) == entry['sha256']
        row = {k:item[k] for k in ('sample', 'projection')}
        try:
            rgb = read_rgb(entry['path']); h = infer_pose(rgb, model, crop=True, turns=(0,1,2,3), is_3d=False)
            row['human'] = h; row['turn'] = h['turn']
            row['bbox'] = foreground_bbox(np.ascontiguousarray(np.rot90(rgb, h['turn'])))
            xy, _, mask = compose_query(h['keypoints2d'], h['scores'], np.zeros(17))
            row['mask'] = mask.tolist(); row['xy'] = xy.tolist()
            if mask[5:].sum()<6 or not mask[[5,6,11,12]].all(): row['error'] = 'insufficient Human-Art joints/anchors'
        except Exception as error: row['error'] = str(error)
        save_json(args.out/'human-raw'/f'{row["sample"]}_{row["projection"]}.json', row)
        records.append(row); print('human', len(records), '/80', flush=True)
    save_json(args.out/'human.json', records)
    save_json(args.out/'human-frozen.json', {'human.json': sha256(args.out/'human.json')})


def infer(args):
    verify(args, ('prepared.json', 'human-frozen.json')); assert not (args.out/'predictions.json').exists()
    model = corrected_rtmw(args.z_model); records = []; (args.out/'raw').mkdir(exist_ok=False)
    manifest = read(args.out/'manifest.json'); humans = read(args.out/'human.json')
    assert len(manifest) == len(humans) == 80
    for item, h in zip(manifest, humans):
        assert (item['sample'],item['projection']) == (h['sample'],h['projection'])
        row = {k:item[k] for k in ('sample','projection')}; row['styles'] = {}
        for style in STYLES:
            entry = item['styles'][style]; assert sha256(Path(entry['path'])) == entry['sha256']
            output = {}
            try:
                if 'human' not in h: raise ValueError(h['error'])
                raw = fixed_infer(read_rgb(entry['path']), model, h['turn'], h['bbox'])
                output['raw'] = raw
                if 'error' in h: output['error'] = h['error']
                else: output['mapped'] = features(h, raw)
            except Exception as error: output['error'] = str(error)
            row['styles'][style] = output
        records.append(row); save_json(args.out/'raw'/f'{row["sample"]}_{row["projection"]}.json',row)
        print('depth',len(records),'/80 pairs',flush=True)
    verify(args, ('prepared.json','human-frozen.json'))
    save_json(args.out/'predictions.json',records)
    save_json(args.out/'predicted.json',{'predictions.json':sha256(args.out/'predictions.json')})


def canonical_observation(human, predicted, gt_xy):
    """GT-only scoring transform. This function is never called in inference."""
    if human.get('error') or predicted.get('error'):
        return {'error': human.get('error') or predicted.get('error'), 'signs':[0]*8,
                'all':[False]*8, 'xy_gate':[False]*8, 'oracle_swap':None}
    mask = np.asarray(human['mask'],bool); body = np.flatnonzero(mask[5:])+5
    h = np.asarray(human['human']['keypoints2d']); gt = np.asarray(gt_xy)
    swap = bool(np.linalg.norm(h[body]-gt[LEFT_RIGHT][body],axis=1).mean() <
                np.linalg.norm(h[body]-gt[body],axis=1).mean())
    permutation = LEFT_RIGHT if swap else np.arange(17)
    mapped = predicted['mapped']['hypotheses']['xy_aligned']
    z = np.asarray(mapped['normalized_z'],np.float32)[permutation]
    errors = np.asarray(mapped['xy_error_torso'])[permutation]; mask = mask[permutation]
    delta = z[PAIRS[:,0]]-z[PAIRS[:,1]]
    active = mask[PAIRS].all(1) & (abs(delta)>=.1)
    return {'oracle_swap':swap, 'signs':np.sign(delta).astype(int).tolist(), 'all':active.tolist(),
            'xy_gate':(active & (errors[PAIRS]<=.5).all(1)).tolist()}


def count_cell(signs, truth, active):
    return {'n':int(active.sum()), 'correct':int(((signs==truth)&active).sum()),
            'always_front':int(((truth<0)&active).sum()), 'always_back':int(((truth>0)&active).sum())}


def paired_summary(records, left, right, gate):
    totals = {'left':left,'right':right,'gate':gate,'gt_n':0,'gains':0,'losses':0,
              'images_better':0,'images_worse':0,'images_tied':0,
              'own':{a:dict(n=0,correct=0,always_front=0,always_back=0) for a in (left,right)},
              'common':{a:dict(n=0,correct=0,always_front=0,always_back=0) for a in (left,right)}}
    for row in records:
        truth = np.asarray(row['truth_signs']); valid = np.asarray(row['truth_valid'],bool)
        signs = {a:np.asarray(row['arms'][a]['signs']) for a in (left,right)}
        active = {a:valid & np.asarray(row['arms'][a][gate],bool) for a in (left,right)}
        common = active[left]&active[right]; totals['gt_n'] += int(valid.sum())
        gain = int((common & (signs[left]!=truth)&(signs[right]==truth)).sum())
        loss = int((common & (signs[left]==truth)&(signs[right]!=truth)).sum())
        totals['gains']+=gain; totals['losses']+=loss
        totals['images_better']+=int(gain>loss); totals['images_worse']+=int(gain<loss); totals['images_tied']+=int(gain==loss)
        for scope in ('own','common'):
            for a in (left,right):
                cell = count_cell(signs[a],truth,active[a] if scope=='own' else common)
                for k,v in cell.items(): totals[scope][a][k] += v
    return totals


def evaluate(args):
    verify(args, ('prepared.json','human-frozen.json','predicted.json'))
    assert not (args.out/'results.json').exists()
    humans={(r['sample'],r['projection']):r for r in read(args.out/'human.json')}
    predictions={(r['sample'],r['projection']):r for r in read(args.out/'predictions.json')}
    answers=read(args.out/'answers.json'); assert len(humans)==len(predictions)==80 and len(answers)==40
    rows=[]; desc={}
    for projection in PROJECTIONS:
        for style in STYLES: desc[projection+'_'+style]={'images':40,'failed':0,'body_score_ge03':0,'errors':[]}
    for answer in answers:
        truth=np.asarray(answer['camera_z']); delta=truth[PAIRS[:,0]]-truth[PAIRS[:,1]]
        row={'sample':answer['sample'],'truth_signs':np.sign(delta).astype(int).tolist(),
             'truth_valid':(abs(delta)>=.1).tolist(),'arms':{}}
        for projection in PROJECTIONS:
            human=humans[(answer['sample'],projection)]; pred=predictions[(answer['sample'],projection)]
            for style in STYLES:
                name=projection+'_'+style; value=pred['styles'][style]
                row['arms'][name]=canonical_observation(human,value,answer['projections'][projection])
                if value.get('error'): desc[name]['failed']+=1
                if 'raw' in value: desc[name]['body_score_ge03']+=int((np.asarray(value['raw']['scores'])[5:]>=.3).sum())
                if 'mapped' in value: desc[name]['errors'].append(min(value['mapped']['mean_direct'],value['mapped']['mean_swapped']))
        rows.append(row)
    comparisons=[]
    for left,right in [('ortho_'+s,'perspective_'+s) for s in STYLES]+[(p+'_rough',p+'_solid') for p in PROJECTIONS]:
        for gate in ('all','xy_gate'): comparisons.append(paired_summary(rows,left,right,gate))
    for value in desc.values():
        values=value.pop('errors'); value['median_xy_disagreement']=float(np.median(values)) if values else None
    results={'diagnostic_only':True,'retrieval_accuracy':None,'description':desc,'comparisons':comparisons,'records':rows}
    save_json(args.out/'results.json',results)
    print(json.dumps({k:v for k,v in results.items() if k!='records'},indent=2))


def parse():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','human','infer','evaluate'))
    for name in ('out','library-root','human-model','z-model'):p.add_argument('--'+name,type=Path,required=True)
    return p.parse_args()


if __name__=='__main__':
    args=parse();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.depth-projection.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
