"""Independently reconstruct every fixed blend policy from saved cost arrays."""
import argparse
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_refinement_blend import verify,read,select_policy,GROUPS,BASELINES
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import sha256,save_json


def audit(args):
    verify(args)
    policy=read(args.out/'protocol.json')
    rows=read(args.out/'rankings.json')
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    original={r['image']:r for r in read(args.refinement_out/'rankings.json')}
    poses=load_library(args.library_root)
    checked=failures=0
    numeric_changes=[];alternative_rows=[];max_arithmetic_delta=0.
    for row in rows:
        source=original[row['image']]
        assert row['prediction_sha256']==source['prediction_sha256']
        if source.get('error'):
            assert row.get('error') and all(a not in row for a in policy['policies'])
            failures+=1;alternative_rows.append(row);continue
        assert row['shared_xy_sha256']==source['shared_xy_sha256'] and row['shared_mask']==source['shared_mask']
        path=args.refinement_out/'costs'/(Path(row['image']).stem+'.npz')
        assert sha256(path)==row['costs_sha256']==source['costs_sha256']
        with np.load(path,allow_pickle=False) as data:
            base=data['baseline'].astype(float);fit=data['refined'].astype(float)
        first=int(np.argmin(base))
        alternative_row=dict(row)
        for arm,(alpha,keep) in policy['policies'].items():
            # Replay the frozen arithmetic order. A mathematically equivalent
            # expansion can split exact float ties at ~1e-16; audit that too.
            score=np.add(np.multiply(base,1-alpha),np.multiply(fit,alpha))
            order=np.argsort(score,kind='stable').tolist()
            if keep:order=[first]+[i for i in order if i!=first]
            assert row[arm]==[poses[i].pose_id for i in order[:5]]
            alternative=base+alpha*(fit-base)
            delta=float(np.max(np.abs(score-alternative)))
            max_arithmetic_delta=max(max_arithmetic_delta,delta)
            assert delta<1e-12
            altorder=np.argsort(alternative,kind='stable').tolist()
            if keep:altorder=[first]+[i for i in altorder if i!=first]
            altids=[poses[i].pose_id for i in altorder[:5]]
            alternative_row[arm]=altids
            if row[arm]!=altids:
                numeric_changes.append({'image':row['image'],'arm':arm,
                    'top1_changed':row[arm][0]!=altids[0],'top5_set_changed':set(row[arm])!=set(altids)})
        assert all(row[a]==source[a] for a in ('pooled_117','keepfirst_195'))
        checked+=1
        alternative_rows.append(alternative_row)
    assert len(rows)==len(original)==len({r['image'] for r in rows})==360
    assert checked==357 and failures==3
    answers={r['image']:r for r in read(args.source/'answers.json')}
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in (*BASELINES,*policy['policies'])}
    for row in alternative_rows:
        target=answers[row['image']]
        for arm in counts:
            ids=row.get(arm,[])
            counts[arm][target['group']]['top1']+=bool(ids and ids[0]==target['pose_id'])
            counts[arm][target['group']]['top5']+=target['pose_id'] in ids
    canonical=read(args.out/'results.json')
    eligible=select_policy(counts)
    result={'ranking_sha256':sha256(args.out/'rankings.json'),'all_six_policy_top5_replayed':checked,
            'unchanged_XY_mask_and_baselines':checked,'failures_retained':failures,
            'audit_script_sha256':sha256(Path(__file__)),
            'equivalent_arithmetic_max_delta':max_arithmetic_delta,'equivalent_arithmetic_order_changes':numeric_changes,
            'equivalent_arithmetic_counts':counts,'all_counts_same_under_alternative_arithmetic':counts==canonical['counts'],
            'selection_same_under_alternative_arithmetic':(eligible[0] if eligible else None)==canonical['selected']}
    save_json(args.out/'audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','refinement-out','out','library-root','model'):
        p.add_argument('--'+name,type=Path,required=True)
    audit(p.parse_args())
