import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.confirm_refinement_blend import success_gate,ARMS,GROUPS,SELECTED


def test_confirmation_gate_requires_near_comparison_and_both_baseline_protections():
    good={g:{'counts':{a:{'top1':20,'top5':60 if a!=SELECTED else 70} for a in ARMS},
        'selected_vs_near':{'delta':.1,'paired_bootstrap_95ci':[.01,.2],'exact_mcnemar_p':.01}} for g in GROUPS}
    assert success_gate(good)
    for g in (GROUPS[0],GROUPS[2]):
        for base in ARMS[:2]:
            bad=copy.deepcopy(good);bad[g]['counts'][base]['top5']=71
            assert not success_gate(bad)
    for g in GROUPS:
        bad=copy.deepcopy(good);bad[g]['counts'][SELECTED]['top1']=21
        assert not success_gate(bad)
    for field,value in [('delta',.04),('paired_bootstrap_95ci',[0,.2]),('exact_mcnemar_p',.05)]:
        bad=copy.deepcopy(good);bad[GROUPS[1]]['selected_vs_near'][field]=value
        assert not success_gate(bad)


if __name__=='__main__':
    test_confirmation_gate_requires_near_comparison_and_both_baseline_protections()
    print('PASS test_confirmation_gate_requires_near_comparison_and_both_baseline_protections')
