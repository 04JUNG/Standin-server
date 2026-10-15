import json
from types import SimpleNamespace
import numpy as np
import pytest

from src.part_laterality import explicit_sides,agreement,verify_mirror_geometry,SWAP,MirrorSelector


def description(text,observations=()):return dict(description=text,observations=list(observations))


def test_explicit_roles_are_anatomical_and_side_specific():
    result=explicit_sides(description('오른팔을 앞으로 굽혀 손을 펼치고 왼팔은 아래로 내린다.'),'upper')
    assert result['facts']['right.location']=='front' and result['facts']['left.location']=='down'
    assert result['facts']['right.elbow']=='bent'
    assert 'left.elbow' not in result['facts']


@pytest.mark.parametrize('text',('한 손을 앞으로 뻗음','양팔을 들어 올림','왼손이 가려 잘 안 보임','화면 오른팔을 내림'))
def test_ambiguous_symmetric_or_screen_relative_facts_abstain(text):
    assert explicit_sides(description(text),'upper')['facts']=={}


def test_observation_conflicts_abstain_and_description_does_not_override():
    result=explicit_sides(description('오른팔은 위로 든다',('오른팔을 아래로 내림','오른팔을 얼굴 근처에 둠')),'upper')
    assert 'right.location' not in result['facts'] and 'right.location' in result['ambiguous_fields']


def test_hand_open_does_not_mean_elbow_is_straight_and_parenthetical_screen_side_is_ignored():
    result=explicit_sides(description('왼팔(화면상 오른팔)을 앞으로 굽혀 손바닥을 펴고 오른팔은 내린다'),'upper')
    assert result['facts']['left.elbow']=='bent' and result['facts']['right.location']=='down'
    assert 'right.elbow' not in result['facts']


def test_lower_leg_rules_do_not_infer_arm_facts():
    result=explicit_sides(description('오른무릎은 구부리고 왼다리는 곧게 편다'),'lower')
    assert result['facts']['right.knee']=='bent' and result['facts']['left.knee']=='straight'
    assert not explicit_sides(description('오른무릎은 구부린다'),'upper')['facts']


def points():
    p=np.zeros((17,3));p[5]=[-.2,1,0];p[6]=[.2,1,0];p[11]=[-.1,0,0];p[12]=[.1,0,0]
    p[7]=[-.3,.6,0];p[9]=[-.3,.1,0];p[8]=[.4,.6,.3];p[10]=[.4,.9,.5]
    p[13]=[-.1,-.5,0];p[15]=[-.1,-1,0];p[14]=[.1,-.5,0];p[16]=[.1,-1,0]
    return p


def test_mirror_proof_requires_reflection_and_anatomical_swap_not_filename():
    a=points();b=a[SWAP].copy();b[:,0]*=-1;s=np.ones(17)
    assert verify_mirror_geometry(a,s,b,s)['verified']
    b[9,2]+=.2
    assert not verify_mirror_geometry(a,s,b,s)['verified']
    assert not verify_mirror_geometry(a,np.zeros(17),a,s)['verified']


def test_mirror_choice_requires_fewer_conflicts_and_a_positive_known_match():
    rows=[dict(pose_id='a',upper=description('오른팔을 앞으로 굽히고 왼팔은 내림')),
          dict(pose_id='a_mirror',upper=description('왼팔을 앞으로 굽히고 오른팔은 내림'))]
    selector=MirrorSelector(SimpleNamespace(records=rows))
    selector.pair=lambda pid:('a' if pid=='a_mirror' else 'a_mirror',{'verified':True})
    query=explicit_sides(description('오른팔 앞으로, 왼팔 아래로'),'upper')
    pid,report=selector.choose('a_mirror','upper',query)
    assert pid=='a' and report['swapped'] and not report['after']['conflicts']
    pid,report=selector.choose('a','upper',query)
    assert pid=='a' and not report['swapped']
    pid,report=selector.choose('a_mirror','upper',explicit_sides(description('양팔을 들어 올림'),'upper'))
    assert pid=='a_mirror' and report['selection'].startswith('abstain')


def test_missing_or_quarantined_mirror_does_not_create_a_new_pose():
    rows=[dict(pose_id='a_mirror',upper=description('왼팔을 앞으로 굽히고 오른팔은 내림'))]
    selector=MirrorSelector(SimpleNamespace(records=rows))
    pid,report=selector.choose('a_mirror','upper',explicit_sides(description('오른팔 앞으로, 왼팔 아래로'),'upper'))
    assert pid=='a_mirror' and not report['swapped'] and not report['mirror_proof']['verified']


def test_index_selects_mirror_before_family_collapse_without_family_rank_bonus(corpus,monkeypatch):
    from tests.test_natural_parts import Encoder,part
    from src.natural_parts import NaturalPartIndex
    bundle,path=corpus;data=json.loads(path.read_text())
    for r in data['poses']:
        if r['pose_id']=='a':r['upper']=part('오른팔 앞으로 굽힘, 왼팔 아래로 내림')
        if r['pose_id']=='a_mirror':r['upper']=part('왼팔 앞으로 굽힘, 오른팔 아래로 내림')
    path.write_text(json.dumps(data))
    monkeypatch.setattr(MirrorSelector,'pair',lambda self,pid:('a',{'verified':True}))
    index=NaturalPartIndex(bundle,path,Encoder());index.vectors['upper']=np.array([[.9,0],[1.,0],[.1,0]])
    index.encoder.encode=lambda _:np.array([[1.,0]])
    query=part('오른팔 앞으로 굽힘, 왼팔 아래로 내림')
    index.side_conditions=False;index.mirror_selector=None
    before=index.search(query,'upper')['hits']
    index.side_conditions=True;index.mirror_selector=MirrorSelector(index)
    after=index.search(query,'upper')['hits']
    assert before[0]['pose_id']=='a_mirror' and after[0]['pose_id']=='a'
    assert after[0]['score']==before[0]['score'] and after[0]['variant_text_score']<after[0]['score']
    assert [h['family']for h in before]==[h['family']for h in after]


# Reuse the exact DB/BVH-bound sidecar fixture rather than a permissive index.
from tests.test_natural_parts import corpus
