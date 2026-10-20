import sys
import copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.confirm_camera_regularization import energy_stats,success_gate,GROUPS,BASE,SELECTED


def test_energy_bootstrap_pairs_families_and_is_deterministic():
    b=np.arange(1,121,dtype=float);s=b*.75
    x=energy_stats(b,s)
    assert x==energy_stats(b,s) and x['families']==120
    assert x['ratio']==.75 and np.allclose(x['family_bootstrap_95ci'],.75)


def test_gate_protects_every_group_and_energy_interval():
    groups={g:{'counts':{a:{'top1':40,'top5':80} for a in (BASE,SELECTED)},
               'selected_vs_baseline':{'paired_bootstrap_95ci':[-.02,.02]}} for g in GROUPS}
    energy={'ratio':.8,'family_bootstrap_95ci':[.7,.9]}
    assert success_gate(groups,energy)
    for g in GROUPS:
        bad=copy.deepcopy(groups);bad[g]['counts'][SELECTED]['top5']-=1
        assert not success_gate(bad,energy)
        bad=copy.deepcopy(groups);bad[g]['selected_vs_baseline']['paired_bootstrap_95ci'][0]=-.051
        assert not success_gate(bad,energy)
    assert not success_gate(groups,{'ratio':.8,'family_bootstrap_95ci':[.7,1.]})


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
