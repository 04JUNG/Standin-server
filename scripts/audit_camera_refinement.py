"""Replay cost-file ordering and independently check selected fitted cameras."""
import argparse
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_camera_refinement import verify,read,ARMS
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import sha256,save_json
from scripts.confirm_camera_fit import geometry_hash
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import Camera,project,normalize_batch


def audit(args):
    verify(args)
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    poses=load_library(args.library_root)
    assert geometry_hash(poses)==read(args.source/'protocol.json')['hashes']['library_geometry']
    rows=read(args.out/'rankings.json')
    old={r['image']:r for r in read(args.source/'rankings.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    permutation=np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])
    checked=failures=physical=0
    maxerror=0.
    for row in rows:
        name=row['image']
        assert sha256(args.source/'images'/name)==manifest[name]['sha256']
        path=args.source/'predictions'/(Path(name).stem+'.json')
        assert sha256(path)==row['prediction_sha256']
        query=valid_query(read(path))
        if query is None:
            assert row.get('error') and old[name].get('error')
            assert all(a not in row for a in ARMS)
            failures+=1;continue
        xy,mask=query;q=normalize_batch(xy)
        assert mask.tolist()==row['shared_mask']
        costpath=args.out/'costs'/(Path(name).stem+'.npz')
        assert sha256(costpath)==row['costs_sha256']
        with np.load(costpath,allow_pickle=False) as data:
            base,coarse,fitted,params=[data[k] for k in ('baseline','coarse','refined','parameters')]
        assert base.shape==coarse.shape==fitted.shape==(len(poses),)
        assert params.shape==(len(poses),3)
        assert np.all(np.isfinite(fitted)) and np.all(fitted>=0)
        assert np.all(fitted<=coarse+1e-6) and np.all(coarse<=base+1e-6)
        bo=np.argsort(base,kind='stable').tolist()
        co=np.argsort(coarse,kind='stable').tolist()
        fo=np.argsort(fitted,kind='stable').tolist()
        first=bo[0]
        orders={'pooled_117':bo,'keepfirst_195':[first]+[i for i in co if i!=first],
                'refined':fo,'keepfirst_refined':[first]+[i for i in fo if i!=first]}
        for arm,order in orders.items():
            assert row[arm]==[poses[i].pose_id for i in order[:5]]
        assert all(row[a]==old[name][a] for a in ARMS[:2])
        # Every output candidate plus a fixed spread of other gallery entries.
        selected=set(fo[:5]+bo[:5]+list(range(0,len(poses),400)))
        for i in selected:
            yaw,pitch,inv=map(float,params[i])
            assert 0<=yaw<360 and -60<=pitch<=60 and 0<=inv<=1.25
            cam=Camera(yaw,pitch,None if inv==0 else 1./inv)
            projected,valid=project(poses[i].joints,cam)
            assert valid
            p=normalize_batch(projected)
            direct=mask.copy();direct[:5]=False
            altmask=mask[permutation].copy();altmask[:5]=False
            dc=float(np.linalg.norm(p[direct]-q[direct],axis=-1).mean())
            ac=float(np.linalg.norm(p[altmask]-q[permutation][altmask],axis=-1).mean())
            error=min(abs(float(fitted[i])-dc),abs(float(fitted[i])-ac))
            assert error<3e-5,(name,i,error)
            maxerror=max(maxerror,error);physical+=1
        checked+=1
    assert len(rows)==len(old)==360 and checked+failures==360
    verify(args)
    result={'ranking_sha256':sha256(args.out/'rankings.json'),'all_cost_file_hashes_verified':True,
            'candidate_orders_replayed':checked,'failures_retained':failures,
            'independent_scalar_projection_checks':physical,'max_scalar_cost_error':maxerror,
            'baseline_top5_matches':checked,'all_gallery_fit_costs_nonincreasing':True,
            'audit_script_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','out','library-root','model'):
        parser.add_argument('--'+name,type=Path,required=True)
    audit(parser.parse_args())
