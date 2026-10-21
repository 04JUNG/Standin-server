from src.basic_closeup import basic_closeup_plan
from src import basic_closeup
from tests.test_fullbody_routing import engine,decision,evidence
import pytest


@pytest.mark.parametrize('scope', ['head', 'shoulder_head'])
def test_basic_closeups_are_paused_without_changing_normal_routing(scope):
    assert basic_closeup.BASIC_CLOSEUP_ENABLED is False
    assert basic_closeup_plan({'output_scope': scope}, {}) is None
    e=engine();ev=evidence();ev['output_scope']=scope
    d=decision(False);d.semantic.description['pose_relations']=[]
    out=e.search(d,ev)
    assert out['route'] != 'basic_model' and not out.get('basic_model')
    assert e.semantic.calls == 1
    assert out['route'] == 'semantic'


def test_explicit_closeup_skips_search_and_generation_when_experiment_enabled(monkeypatch):
    monkeypatch.setattr(basic_closeup, 'BASIC_CLOSEUP_ENABLED', True)
    e=engine();ev=evidence();ev['keypoints']=ev['keypoints'].tolist();ev['joint_mask']=ev['joint_mask'].tolist()
    ev['output_scope']='shoulder_head'
    d=decision(False);d.semantic.description['pose_relations']=[]
    result=e.search(d,ev)
    assert result['route']=='basic_model' and result['selected']==[]
    assert e.camera.calls==e.semantic.calls==e.generator.calls==0
    assert not result['basic_model']['search_match']


def test_no_body_joint_requirement_but_no_implicit_crop(monkeypatch):
    monkeypatch.setattr(basic_closeup, 'BASIC_CLOSEUP_ENABLED', True)
    assert basic_closeup_plan({'output_scope':'shoulder_head'}, {})
    assert basic_closeup_plan({'output_scope':'half'}, {}) is None
    assert basic_closeup_plan({'output_scope':'shoulder_head'}, {'visible_scope':'half'}) is None
    assert basic_closeup_plan({'output_scope':'shoulder_head'}, {'pose_relations':[{'kind':'touch_face'}]}) is None


def test_eye_roll_is_bounded_and_optional(monkeypatch):
    monkeypatch.setattr(basic_closeup, 'BASIC_CLOSEUP_ENABLED', True)
    p=[[0,0] for _ in range(17)];p[1]=[10,10];p[2]=[30,15]
    ev={'output_scope':'shoulder_head','keypoints':p,'joint_mask':[True]*17}
    assert 0<basic_closeup_plan(ev,{})['head_roll_degrees']<25
    p[2]=[30,100]
    assert basic_closeup_plan(ev,{})['head_roll_source']=='neutral_default'
