from copy import deepcopy
from dataclasses import asdict
import json
import subprocess
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.natural_pose import (NaturalLanguagePoseGenerator, generate_loop,
                              build_natural_generator, VERSION)
from src.natural_pose_schema import validate_proposal
from src.natural_pose_worker import compile_spec, independent_reasons


TAXO = {'vocabulary': {k: {v: v} for k,v in dict(posture='standing', action='wave',
    domain='daily', activity='rest', object='phone', emotion='happy', intent='greeting').items()}}


def proposal():
    return dict(base='st_neutral', upper=None, spec={}, tags=dict(posture=['standing'], action=['wave']),
        body_parts=dict(lower_posture='서기', upper_action='팔 내림', active_hand='none',
                        hands=dict(L='relaxed', R='relaxed'), head_direction='정면'))


def passed(value):
    return dict(passed=True, reasons=[], validation=dict(independent=True), candidate={'bvh': 'test'})


def test_valid_spec_and_no_default_erase():
    raw = proposal(); raw['base'] = 'ch_normal'; raw['spec'] = {'root': {'yaw': 25}}
    spec = compile_spec(validate_proposal(raw, TAXO))
    assert spec['root']['y'] == 53.4 and spec['root']['yaw'] == 25
    assert spec['legs']['L']['to'][2] == 40 and spec['props']


@pytest.mark.parametrize('patch', [
    {'support':'none'}, {'place':'none'}, {'joint_rotations':[]},
    {'support_points':[[0,0]]}, {'partner_support':[]},
    {'root': {'y': float('nan')}}, {'root': {'y': 1e30}},
    {'arms': {'L': {'to':[0,2], 'frame':'c'}}},
    {'arms': {'L': {'to':[0,2,3], 'frame':'O.h'}}},
    {'hands': {'L':'invented_hand'}},
    {'props':[{'kind':'seat', 'min':[0,0,0], 'max':[1,2,3], 'collide':False}]},
    {'props':[{'kind':'seat', 'min':[1,2,3], 'max':[0,2,3]}]},
])
def test_untrusted_spec_rejected(patch):
    raw = proposal(); raw['spec'] = patch
    with pytest.raises(ValidationError):validate_proposal(raw, TAXO)


def test_unknown_taxonomy_rejected():
    raw = proposal(); raw['tags']['action'] = ['invented']
    with pytest.raises(ValueError, match='unknown search'):validate_proposal(raw, TAXO)


def test_unknown_and_incompatible_catalog_rejected():
    raw = proposal(); raw['base'] = '../evil'
    with pytest.raises(ValueError, match='unknown_base'):compile_spec(validate_proposal(raw,TAXO))
    raw = proposal(); raw['upper'] = '../evil'
    with pytest.raises(ValueError, match='incompatible'):compile_spec(validate_proposal(raw,TAXO))


def test_metadata_must_match_actual_hand():
    raw = proposal(); raw['spec'] = {'hands':{'L':'phone'}}
    with pytest.raises(ValueError, match='hand_metadata'):compile_spec(validate_proposal(raw,TAXO))


def test_success_never_auto_release_or_refine():
    result = generate_loop('문장',lambda *args:proposal(),passed,TAXO)
    assert result.status == 'preview_ready' and result.backend == VERSION
    assert len(result.candidates)==1 and not result.candidates[0]['refine_allowed']
    assert not result.to_dict()['release_allowed'] and not result.to_dict()['actual_mesh_validated']


def test_retry_receives_actual_failures_and_previous_spec():
    calls=[]; count=0
    def planner(text, previous, feedback, remaining):
        calls.append((previous,feedback)); return proposal()
    def solver(value):
        nonlocal count
        count+=1
        return passed(value) if count==2 else dict(passed=False,reasons=['R 손 접촉점 미도달 8.5cm'])
    result=generate_loop('턱 괴기',planner,solver,TAXO)
    assert result.status=='preview_ready' and count==2
    assert calls[1][0]==proposal() and calls[1][1]==['R 손 접촉점 미도달 8.5cm']


def test_invalid_schema_never_enters_solver_and_four_attempt_cap():
    calls=[]
    result=generate_loop('test',lambda *args:{},lambda x:calls.append(x),TAXO)
    assert result.status=='rejected' and len(result.diagnostics['attempts'])==4 and not calls


def test_deadline_after_planner_never_calls_solver():
    now=[0.]; calls=[]
    def planner(*args):now[0]=61.;return proposal()
    result=generate_loop('test',planner,lambda x:calls.append(x),TAXO,clock=lambda:now[0])
    assert result.status=='timeout' and not calls and not result.candidates


def test_independent_audit_is_stricter_and_finite():
    clean=dict(bend=90,forearm_roll=0,wrist_flex=0,wrist_rad=0)
    assert not independent_reasons({'L':clean},{'Luarm~chest':.99})
    assert independent_reasons({'L':clean},{'Luarm~chest':1.01})
    assert independent_reasons({'L':clean},{'Lhand~chest':1.51})
    for key,value in [('forearm_roll',89),('wrist_flex',71),('wrist_rad',23),('bend',float('nan'))]:
        assert independent_reasons({'L':{**clean,key:value}}, {})


def test_generator_does_not_call_worker_on_bad_rig(tmp_path,monkeypatch):
    rig=tmp_path/'rig.bvh';rig.write_text('bad')
    tax=tmp_path/'tax.json';tax.write_text('{}')
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('worker must not start'))
    result=NaturalLanguagePoseGenerator(rig,tax).generate_text('팔짱')
    assert result.status=='error' and not result.candidates


def test_worker_timeout_does_not_become_success(tmp_path,monkeypatch):
    import src.natural_pose as mod
    monkeypatch.setattr(mod,'file_hash',lambda p:mod.RIG_SHA256)
    def run(*args,**kwargs):
        assert 0 < kwargs['timeout'] <= .1
        raise subprocess.TimeoutExpired('worker',.1)
    monkeypatch.setattr(subprocess,'run',run)
    result=NaturalLanguagePoseGenerator(tmp_path/'rig',tmp_path/'taxonomy',budget_s=.1).generate_text('팔짱')
    assert result.status=='timeout' and not result.candidates


def test_no_key_does_not_enable_generation(monkeypatch):
    monkeypatch.delenv('POSE_GENERATION_ENABLED',raising=False)
    assert build_natural_generator() is None


def test_real_description_contract_to_text(tmp_path, monkeypatch):
    from src.search_routing import PoseDescription
    from src.experimental.text_pose_generation import GenerationResult
    generator=NaturalLanguagePoseGenerator(tmp_path/'rig',tmp_path/'tax')
    calls=[]
    monkeypatch.setattr(generator,'generate_text',lambda text:(calls.append(text) or GenerationResult('rejected')))
    result=generator.generate(PoseDescription(visible=True,certainty='high',search_text='의자에 앉기',body_details='양팔로 팔짱'))
    assert calls==['의자에 앉기. 양팔로 팔짱'] and result.status=='rejected'
