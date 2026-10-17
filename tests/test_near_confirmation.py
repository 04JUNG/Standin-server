import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.confirm_near_camera import success_gate,GROUPS


def test_gate_uses_strong_not_moderate_and_preserves_top1():
    good={g:{'counts':{'pooled_117':{'top1':20,'top5':40},'keepfirst_195':{'top1':20,'top5':50}},
             'near_vs_baseline':{'delta':.1,'paired_bootstrap_95ci':[.01,.19],'exact_mcnemar_p':.01}} for g in GROUPS}
    good['perspective']['near_vs_baseline']['delta']=0.
    assert success_gate(good)
    for g in GROUPS:
        bad=copy.deepcopy(good)
        bad[g]['counts']['keepfirst_195']['top1']=21
        assert not success_gate(bad)
    for g in ('perspective','orthographic_control'):
        bad=copy.deepcopy(good)
        bad[g]['counts']['keepfirst_195']['top5']=39
        assert not success_gate(bad)
    for field,value in [('delta',.04),('paired_bootstrap_95ci',[0,.2]),('exact_mcnemar_p',.05)]:
        bad=copy.deepcopy(good)
        bad['strong_perspective']['near_vs_baseline'][field]=value
        assert not success_gate(bad)


if __name__=='__main__':
    test_gate_uses_strong_not_moderate_and_preserves_top1()
    print('PASS test_gate_uses_strong_not_moderate_and_preserves_top1')
