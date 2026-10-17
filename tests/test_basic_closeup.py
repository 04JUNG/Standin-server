from src.basic_closeup import basic_closeup_plan
from tests.test_fullbody_routing import engine,decision,evidence


def test_explicit_closeup_skips_search_and_generation():
    e=engine();ev=evidence();ev['keypoints']=ev['keypoints'].tolist();ev['joint_mask']=ev['joint_mask'].tolist()
    ev['output_scope']='shoulder_head'
    d=decision(False);d.semantic.description['pose_relations']=[]
    result=e.search(d,ev)
    assert result['route']=='basic_model' and result['selected']==[]
    assert e.camera.calls==e.semantic.calls==e.generator.calls==0
    assert not result['basic_model']['search_match']


def test_no_body_joint_requirement_but_no_implicit_crop():
    assert basic_closeup_plan({'output_scope':'shoulder_head'}, {})
    assert basic_closeup_plan({'output_scope':'half'}, {}) is None
    assert basic_closeup_plan({'output_scope':'shoulder_head'}, {'visible_scope':'half'}) is None
    assert basic_closeup_plan({'output_scope':'shoulder_head'}, {'pose_relations':[{'kind':'touch_face'}]}) is None


def test_eye_roll_is_bounded_and_optional():
    p=[[0,0] for _ in range(17)];p[1]=[10,10];p[2]=[30,15]
    ev={'output_scope':'shoulder_head','keypoints':p,'joint_mask':[True]*17}
    assert 0<basic_closeup_plan(ev,{})['head_roll_degrees']<25
    p[2]=[30,100]
    assert basic_closeup_plan(ev,{})['head_roll_source']=='neutral_default'
