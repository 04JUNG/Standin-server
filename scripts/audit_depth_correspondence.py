"""Scalar audit of model XY mapping, pair masks and independent BVH yaw depth."""
import math
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_depth_correspondence import parse,verify,read,CACHE,ANSWERS,ARMS,GATES,load_library
from scripts.experiment_z_rendered import sha256,save_json

PAIRS=[(7,5),(9,5),(8,6),(10,6),(13,11),(15,11),(14,12),(16,12)]


def audit(args):
    verify(args);assert not (args.out/'audit.json').exists()
    rows=read(args.out/'predictions.json');result=read(args.out/'results.json')
    assert sha256(args.out/'predictions.json')==read(args.out/'predicted.json')['sha256']
    cache=read(CACHE);old={(d,r['image']):r for d in ('synthetic','real') for r in cache[d]['records']}
    answers={r['image']:r for r in read(ANSWERS)};scored={r['image']:r for r in result['records']}
    poses={p.pose_id:p for p in load_library(args.library_root)}
    swap=lambda j:(j+1 if j%2 else j-1) if j>=5 else [0,2,1,4,3][j]
    matched=failed=pairchecks=0;maxdepth=0.
    for row in rows:
        source=old[row['dataset'],row['image']]
        assert sha256(Path(source['path']))==source['sha256']
        if source.get('error'):
            assert row['error']==source['error'] and not row['searchable'];failed+=1;continue
        h=source['human']['keypoints2d'];r=source['rtmw']['keypoints2d'];mask=source['shared_mask']
        torso=math.hypot((h[5][0]+h[6][0]-h[11][0]-h[12][0])/2,(h[5][1]+h[6][1]-h[11][1]-h[12][1])/2)
        body=[j for j in range(5,17) if mask[j]]
        direct=sum(math.dist(h[j],r[j]) for j in body)/len(body)/torso
        reverse=sum(math.dist(h[j],r[swap(j)]) for j in body)/len(body)/torso
        assert row['swapped']==(reverse<direct)
        assert math.isclose(row['mean_direct'],direct,abs_tol=1e-12) and math.isclose(row['mean_swapped'],reverse,abs_tol=1e-12)
        for arm in ARMS:
            mapping=[swap(j) if arm=='xy_aligned' and row['swapped'] else j for j in range(17)]
            raw=source['rtmw']['z_meters'];center=(raw[mapping[11]]+raw[mapping[12]])/2
            z=[(raw[k]-center)/.5 if mask[j] else 0. for j,k in enumerate(mapping)]
            saved=row['hypotheses'][arm];difference=max(abs(a-b) for a,b in zip(z,saved['normalized_z']))
            maxdepth=max(maxdepth,difference);assert difference<1e-6
            errors=[math.dist(h[j],r[k])/torso for j,k in enumerate(mapping)]
            assert np.allclose(errors,saved['xy_error_torso'],atol=1e-12,rtol=1e-12)
            assert saved['scores']==[source['rtmw']['scores'][k] for k in mapping]
            for i,(a,b) in enumerate(PAIRS):
                diff=z[a]-z[b];use=bool(mask[a] and mask[b] and abs(diff)>=.1)
                assert saved['signs'][i]==int(np.sign(diff)) and saved['all'][i]==use
                assert saved['xy_gate'][i]==bool(use and errors[a]<=.5 and errors[b]<=.5);pairchecks+=1
        matched+=1
        if row['dataset']!='synthetic':continue
        answer=answers[row['image']];gt=answer['gt_projected_xy']
        human_swap=sum(math.dist(h[j],gt[swap(j)]) for j in body)<sum(math.dist(h[j],gt[j]) for j in body)
        record=scored[row['image']];assert record['oracle_human_gt_swap']==human_swap
        xyz=poses[answer['pose_id']].joints.astype(float)
        hip=(xyz[11]+xyz[12])/2;shoulder=(xyz[5]+xyz[6])/2;scale=math.dist(hip,shoulder)
        theta=math.radians(answer['yaw'])
        z=[(-math.sin(theta)*(p[0]-hip[0])+math.cos(theta)*(p[2]-hip[2]))/scale for p in xyz]
        for frame in ('semantic','observed_human'):
            tz=[z[swap(j)] if frame=='observed_human' and human_swap else z[j] for j in range(17)]
            deltas=[tz[a]-tz[b] for a,b in PAIRS];sign=np.sign(deltas).astype(int)
            truth_valid=np.abs(deltas)>=.1
            for gate in GATES:
                eligible={a:truth_valid&np.array(row['hypotheses'][a][gate],bool) for a in ARMS}
                correct={a:sign==row['hypotheses'][a]['signs'] for a in ARMS}
                common=eligible['direct']&eligible['xy_aligned'];cell=record['frames'][frame][gate]
                for scope in ('own','common'):
                    for arm in ARMS:
                        use=eligible[arm] if scope=='own' else common
                        expected={'n':int(use.sum()),'correct':int((correct[arm]&use).sum()),
                                  'always_front':int(((sign<0)&use).sum()),'always_back':int(((sign>0)&use).sum())}
                        assert cell[scope][arm]==expected
                assert cell['gains']==int((~correct['direct']&correct['xy_aligned']&common).sum())
                assert cell['losses']==int((correct['direct']&~correct['xy_aligned']&common).sum())
    assert matched+failed==len(rows)==50 and len(scored)==40
    for frame,gates in result['depth_order'].items():
        for gate,cell in gates.items():
            cells=[r['frames'][frame][gate] for r in scored.values()]
            for scope in ('own','common'):
                for arm in ARMS:
                    for key,n in cell[scope][arm].items():assert n==sum(c[scope][arm][key] for c in cells)
            for key in ('gains','losses'):assert cell[key]==sum(c[key] for c in cells)
    for dataset,description in result['descriptive'].items():
        allrows=[r for r in rows if r['dataset']==dataset];valid=[r for r in allrows if r['searchable']]
        assert description['images']==len(allrows) and description['searchable']==len(valid)
        assert description['mapped_swaps']==sum(r['swapped'] for r in valid)
        assert description['direct_xy_error_median']==float(np.median([r['mean_direct'] for r in valid]))
        assert description['aligned_xy_error_median']==float(np.median([min(r['mean_direct'],r['mean_swapped']) for r in valid]))
        for arm in ARMS:
            for gate in GATES:assert description['used_pairs'][arm][gate]==sum(sum(r['hypotheses'][arm][gate]) for r in valid)
    assert result['oracle_human_gt_swaps']==sum(r['oracle_human_gt_swap'] for r in scored.values())
    verify(args)
    report={'records_retained':50,'mapped_records_checked':matched,'failures_retained':failed,
            'query_pair_masks_checked':pairchecks,'independent_truth_geometry_images':40,
            'max_scalar64_vs_float32_depth_error':maxdepth,'all_counts_and_common_pair_comparisons_checked':True,
            'predictions_sha256':sha256(args.out/'predictions.json'),'results_sha256':sha256(args.out/'results.json'),
            'auditor_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',report);print(report,flush=True)


if __name__=='__main__':audit(parse())
