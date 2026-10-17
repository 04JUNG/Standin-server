import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.confirm_contour_rerank import success_gate,BASE,SELECTED,GROUPS
from scripts.audit_contour_confirmation import independent_stats


def test_gate_requires_effect_uncertainty_and_control_protection():
    groups={g:{'counts':{BASE:dict(top1=10,top5=50),SELECTED:dict(top1=10,top5=56)},
               'comparison':{'delta':.05,'paired_bootstrap_95ci':[.01,.1],'exact_mcnemar_p':.03}} for g in GROUPS}
    assert success_gate(groups)
    for field,value in (('delta',.049),('paired_bootstrap_95ci',[0,.1]),('exact_mcnemar_p',.05)):
        changed=copy.deepcopy(groups);changed[GROUPS[1]]['comparison'][field]=value
        assert not success_gate(changed)
    for g in GROUPS:
        changed=copy.deepcopy(groups);changed[g]['counts'][SELECTED]['top1']=9;assert not success_gate(changed)
    for g in (GROUPS[0],GROUPS[2]):
        changed=copy.deepcopy(groups);changed[g]['counts'][SELECTED]['top5']=49;assert not success_gate(changed)


def test_failed_records_remain_in_paired_statistics():
    rows=[{BASE+'_rank':None,SELECTED+'_rank':1} for _ in range(6)]
    rows += [{BASE+'_rank':None,SELECTED+'_rank':None,'error':'failed'} for _ in range(114)]
    out=independent_stats(rows)
    assert out['delta']==.05 and out['gains']==6 and out['losses']==0 and out['exact_mcnemar_p']==.03125
    assert out['paired_bootstrap_95ci'][0]>0


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
