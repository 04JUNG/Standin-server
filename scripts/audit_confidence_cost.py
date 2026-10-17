"""Independent weighted residual arithmetic and scalar camera reprojection."""
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_confidence_cost import parse, verify, read, BASE, STRENGTHS, GROUPS
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json, sha256
from scripts.confirm_camera_fit import geometry_hash
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import Camera, project, normalize_batch
from standin_eval.camera_refinement import normalized_joints, project_parameters


def audit(args):
    verify(args)
    assert not (args.out / 'audit.json').exists()
    rows = read(args.out / 'rankings.json')
    assert sha256(args.out / 'rankings.json') == read(args.out / 'ranked.json')['sha256']
    poses = load_library(args.library_root)
    assert geometry_hash(poses) == read(args.source / 'protocol.json')['hashes']['library_geometry']
    points = normalized_joints(np.stack([p.joints for p in poses]))
    originals = {r['image']: r for r in read(args.refinement_out / 'rankings.json')}
    blend = {r['image']: r for r in read(args.blend_out / 'rankings.json')}
    manifest = {r['image']: r for r in read(args.source / 'manifest.json')}
    checked = failed = scalar = 0; max_delta = 0.; control_delta = 0.; uniform_arithmetic_delta = 0.
    for row in rows:
        name = row['image']; path = args.source / 'predictions' / (Path(name).stem + '.json')
        assert sha256(path) == row['prediction_sha256']
        assert sha256(args.source / 'images' / name) == manifest[name]['sha256']
        pred = read(path); query = valid_query(pred)
        if query is None:
            assert row['error'] == originals[name]['error'] and all(a not in row for a in STRENGTHS)
            failed += 1; continue
        xy, mask = query; q = normalize_batch(xy)
        assert hashlib.sha256(xy.tobytes()).hexdigest() == row['shared_xy_sha256']
        assert mask.tolist() == row['shared_mask']
        oldpath = args.refinement_out / 'costs' / (Path(name).stem + '.npz')
        assert sha256(oldpath) == row['original_costs_sha256'] == originals[name]['costs_sha256']
        with np.load(oldpath, allow_pickle=False) as d:
            base, fit, params = [d[k] for k in ('baseline', 'refined', 'parameters')]
        path = args.out / 'costs' / (Path(name).stem + '.npz')
        assert sha256(path) == row['costs_sha256']
        with np.load(path, allow_pickle=False) as d:
            uniform, weighted = d['uniform'], d['weighted']
        features, valid = project_parameters(points, params); assert valid.all()
        scores = pred['scores']
        # No production confidence helper; explicit slot mapping and weighted accumulation.
        uniform_expected = []
        for swapped in (False, True):
            mapping = {j: (j+1 if j%2 else j-1) if swapped else j for j in range(5,17)}
            parts = [(j,k) for j,k in mapping.items() if mask[k]]
            residuals = [np.sqrt(np.sum((features[:,j]-q[k])**2,axis=1)) for j,k in parts]
            uniform_expected.append(np.stack(residuals,axis=1).mean(axis=1))
        expected = []
        for swapped in (False, True):
            total = np.zeros(len(poses)); denom = 0.
            for j in range(5,17):
                k = (j+1 if j%2 else j-1) if swapped else j
                if not mask[k]: continue
                w = .25+.75*min(1.,max(0.,(scores[k]-.3)/.4))
                total += np.linalg.norm(features[:,j]-q[k],axis=1).astype(float)*w; denom += w
            expected.append(total/denom)
        choice = np.argmin(abs(np.stack(uniform_expected)-fit),axis=0)
        cols = np.arange(len(poses))
        assert np.allclose(weighted,np.stack(expected)[choice,cols],atol=1e-12,rtol=1e-12)
        recomputed_uniform = np.stack(uniform_expected)[choice,cols]
        uniform_arithmetic_delta = max(uniform_arithmetic_delta,float(np.max(abs(uniform-recomputed_uniform))))
        assert np.allclose(uniform,recomputed_uniform,atol=2e-6,rtol=1e-6)
        control_delta = max(control_delta,float(np.max(np.abs(uniform-fit))))
        assert np.allclose(uniform,fit,atol=2e-5,rtol=0)
        selected = set(range(0,len(poses),600)); first = int(np.argmin(base))
        for arm,alpha in ((BASE,0.),*STRENGTHS.items()):
            cost = .25*base.astype(float)+.75*((1-alpha)*fit.astype(float)+alpha*weighted)
            indices = np.argsort(cost,kind='stable').tolist(); indices = [first]+[i for i in indices if i!=first]
            assert row[arm] == [poses[i].pose_id for i in indices[:5]]
            selected.update(indices[:5])
        assert row[BASE] == blend[name][BASE]
        for i in selected:
            yaw,pitch,inv = map(float,params[i])
            xy3,ok = project(poses[i].joints,Camera(yaw,pitch,None if inv==0 else 1/inv)); assert ok
            projected = normalize_batch(xy3); costs = []
            for swapped in (False,True):
                num=den=0.
                for j in range(5,17):
                    k=(j+1 if j%2 else j-1) if swapped else j
                    if not mask[k]: continue
                    w=.25+.75*min(1.,max(0.,(scores[k]-.3)/.4))
                    num+=float(np.linalg.norm(projected[j]-q[k]))*w; den+=w
                costs.append(num/den)
            delta=abs(costs[choice[i]]-weighted[i]); assert delta<3e-5
            max_delta=max(max_delta,delta); scalar+=1
        checked+=1
    assert checked+failed==len(rows)==360
    answers={r['image']:r for r in read(args.source/'answers.json')}
    result=read(args.out/'results.json')
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in (BASE,*STRENGTHS)}
    changes={a:{g:{'gains':0,'losses':0} for g in GROUPS} for a in STRENGTHS}; failures={g:0 for g in GROUPS}
    for row in rows:
        t=answers[row['image']]; g=t['group']; pid=t['pose_id']; failures[g]+=bool(row.get('error'))
        for arm in counts:
            ids=row.get(arm,[]); counts[arm][g]['top1']+=bool(ids and ids[0]==pid); counts[arm][g]['top5']+=pid in ids
            if arm!=BASE:
                before=pid in row.get(BASE,[]); after=pid in ids
                changes[arm][g]['gains']+=after and not before; changes[arm][g]['losses']+=before and not after
    assert counts==result['counts'] and changes==result['top5_changes'] and failures==result['failures']
    eligible=[a for a in STRENGTHS if counts[a]['strong_perspective']['top5']>=counts[BASE]['strong_perspective']['top5']+3
              and all(counts[a][g][k]>=counts[BASE][g][k] for g in GROUPS for k in ('top1','top5'))]
    eligible.sort(key=lambda a:(-counts[a]['strong_perspective']['top5'],-sum(counts[a][g]['top5'] for g in GROUPS),STRENGTHS[a]))
    assert result['eligible']==eligible and result['selected']==(eligible[0] if eligible else None)
    verify(args)
    report={'inputs_checked':360,'all_gallery_costs_and_rankings_checked':checked,'failures_retained':failed,
            'independent_scalar_projections':scalar,'maximum_scalar_error':max_delta,
            'maximum_uniform_control_delta':control_delta,'maximum_uniform_arithmetic_delta':uniform_arithmetic_delta,'outcome_recomputed':True,
            'ranking_sha256':sha256(args.out/'rankings.json'),'audit_script_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',report); print(report,flush=True)


if __name__=='__main__': audit(parse())
