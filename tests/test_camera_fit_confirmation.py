import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.confirm_camera_fit import paired_comparison


def test_failures_stay_in_paired_denominator():
    rows=[{'new_rank':1,'old_rank':10},{'new_rank':None,'old_rank':None},
          {'new_rank':8,'old_rank':9},{'new_rank':2,'old_rank':1}]
    result=paired_comparison(rows,'new','old')
    assert result['gains']==1 and result['losses']==0 and result['delta']==.25
    assert result['exact_mcnemar_p']==1.


def test_exact_discordant_test_and_bootstrap_are_deterministic():
    rows=[{'new_rank':1,'old_rank':6} for _ in range(10)]
    rows += [{'new_rank':6,'old_rank':1} for _ in range(2)]
    first=paired_comparison(rows,'new','old')
    assert first==paired_comparison(rows,'new','old')
    assert abs(first['exact_mcnemar_p']-158/4096)<1e-12
    reverse=paired_comparison(rows,'old','new')
    assert reverse['exact_mcnemar_p']==first['exact_mcnemar_p']
    assert reverse['delta']==-first['delta']


def test_no_changes_do_not_pass():
    rows=[{'new_rank':1,'old_rank':1},{'new_rank':10,'old_rank':10}]
    result=paired_comparison(rows,'new','old')
    assert result['delta']==0 and result['paired_bootstrap_95ci']==[0.,0.]
    assert result['exact_mcnemar_p']==1


if __name__=='__main__':
    for name,f in sorted(list(globals().items())):
        if name.startswith('test_'):f();print('PASS',name)
