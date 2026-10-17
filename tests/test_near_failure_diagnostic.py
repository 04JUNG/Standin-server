import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_near_camera_failures import failure_bucket,rank_of,hit,pooled_cost,summarize,GROUPS
from standin_eval.perspective_search import score_camera_index
from standin_eval.dual_label_search import alternate_labels


def test_failure_buckets_are_ordered_and_oracles_not_observed_success():
    row={'keepfirst_195_rank':10,'raw_195_rank':9,'oracle_xy_grid195_rank':2,'oracle_xy_exact_camera_rank':1}
    assert failure_bucket(row)=='exact_xy_recovers_on_same_grid'
    row['oracle_xy_grid195_rank']=8
    assert failure_bucket(row)=='exact_xy_needs_source_camera'
    row['oracle_xy_exact_camera_rank']=6
    assert failure_bucket(row)=='not_recovered_even_with_both_oracles'
    row['raw_195_rank']=5
    assert failure_bucket(row)=='keepfirst_displacement'
    row['keepfirst_195_rank']=1
    assert failure_bucket(row)=='observed_success'
    row['error']='missing anchors'
    assert failure_bucket(row)=='extraction_failure'


def test_rank_lookup_and_missing_are_failures():
    assert rank_of([4,0,2],2)==3
    assert rank_of([4,0,2],1) is None
    assert not hit(None) and not hit(6) and hit(5) and not hit(2,1)


def test_oracle_scoring_preserves_observation_mask():
    rng=np.random.default_rng(12)
    q=rng.normal(size=(17,2)).astype(np.float32)
    mask=np.ones(17,dtype=bool);mask[9]=False
    features=np.stack([q,q+1])[None,...]
    valid=np.ones((1,2),dtype=bool)
    index=(features,valid)
    alt,altmask=alternate_labels(q,mask)
    expected=np.minimum(score_camera_index(index,q,mask),score_camera_index(index,alt,altmask))
    assert np.allclose(pooled_cost(index,q,mask),expected)
    assert not mask[9] and mask.sum()==16


def test_summary_keeps_failed_extractions_out_of_oracle_successes():
    arms=('pooled_117','keepfirst_195','raw_195','observed_xy_exact_camera','oracle_xy_grid195','oracle_xy_exact_camera')
    rows=[]
    for group in GROUPS:
        for i in range(120):
            row={'group':group,'family_id':str(i)}
            if i<3:
                row['error']='extraction'
            else:
                row.update({a+'_rank':1 for a in arms})
                row['label_best_xy_error_torso']=.2
            rows.append(row)
    result=summarize(rows)
    for group in GROUPS:
        assert result[group]['n']==120 and result[group]['searchable']==117
        assert result[group]['counts']['oracle_xy_exact_camera']['top5']==117
        assert result[group]['failure_buckets']=={'extraction_failure':3,'observed_success':117}


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn();print('PASS',name)
