import copy
import json
from itertools import permutations

import numpy as np
import pytest

pytest.importorskip('scipy')
from src.bvh import channel_starts, fk, parse_bvh
from src.experimental.text_pose_generation import (
    GenerationResult, TextConstraintPoseGenerator, _generate_frame, after_no_result,
    compile_plan, two_bone, validate_generated,
)
from src.pipeline import Pipeline
from src.library import build_synthetic_index
from src.search_routing import DescribeOutcome, PoseDescription
from src.config import CFG
from src.vlm.client import MockVLMClient
from tests.test_part_composition import rig
from tests.test_search_routing import _Img, _RoughPose, _StubSearcher, _signature


def description(**changes):
    args=dict(postures=['sitting'],actions=['fold_arms'],certainty='high',
        body_details='가슴 앞에서 양팔을 교차하고 무릎을 직각으로 굽혀 앉아 있다.',
        search_text='팔짱을 끼고 무릎을 굽혀 앉은 자세',body_observations={'torso':'upright','knees':'bent'})
    args.update(changes);return PoseDescription(**args)


def serialize_rig(joints,motion):
    children={i:[] for i in range(len(joints))}
    for i,j in enumerate(joints):
        if j[1]!=-1:children[j[1]].append(i)
    lines=['HIERARCHY']
    def emit(i,depth):
        name,parent,offset,channels,end=joints[i];pad='  '*depth
        lines.extend([pad+('ROOT ' if parent==-1 else 'JOINT ')+name,pad+'{',pad+'  OFFSET '+' '.join(map(str,offset)),pad+'  CHANNELS '+str(len(channels))+' '+' '.join(channels)])
        for child in children[i]:emit(child,depth+1)
        lines.append(pad+'}')
    emit(0,0);lines.extend(['MOTION','Frames: 1','Frame Time: 0.033333',' '.join(map(str,motion))]);return '\n'.join(lines)+'\n'


@pytest.mark.parametrize('order',[''.join(p) for p in permutations('XYZ')])
def test_generated_fold_seated_preserves_bones_with_world_root_offset(order):
    j,_=rig(scale=100,prefix='mixamorig:',order=order);j[0][2]=np.array([12.,94.,-7.]);saved=copy.deepcopy(j)
    plan=compile_plan(description());frame,_=_generate_frame(j,plan,0);check=validate_generated(j,frame,plan)
    assert check['kinematic_probe_pass'],check['reasons']
    assert check['relations']['fold_geometry'] is True
    assert check['knee_deg']==pytest.approx([90,90],abs=1e-4)
    assert check['bone_length_delta_torso']<1e-6
    assert check['actual_mesh_validated'] is check['release_allowed'] is False
    for actual,before in zip(j,saved):np.testing.assert_array_equal(actual[2],before[2])
    starts=channel_starts(j)
    for i,joint in enumerate(j):
        for k,ch in enumerate(joint[3]):
            if ch.endswith('position') and joint[1]!=-1:assert frame[starts[i]+k]==0
            if joint[0].endswith('Hand') and ch.endswith('rotation'):assert frame[starts[i]+k]==0


@pytest.mark.parametrize('posture',['standing','sitting'])
def test_v_arms_supported_with_simple_symmetric_posture(posture):
    d=description(postures=[posture],actions=[],search_text=f'{posture} with arms in a V',
        body_details='양팔을 V자로 넓게 벌린 자세',body_observations={})
    j,_=rig();plan=compile_plan(d);f,_=_generate_frame(j,plan,0);result=validate_generated(j,f,plan)
    assert result['kinematic_probe_pass'],result['reasons']
    assert result['relations']['v_arms'] is True


@pytest.mark.parametrize('changes',[
    {'certainty':'low'},{'certainty':None},{'visible':False},
    {'search_text':'팔짱을 끼지 않은 자세'},
    {'search_text':'팔짱은 불확실하다'},
    {'search_text':'양팔 V자와 팔짱을 동시에 한 자세'},
    {'postures':['sitting','standing']},{'objects':['sword']},{'actions':['fold_arms','run']},
    {'search_text':'한쪽 무릎을 접어 공중에서 팔짱을 낀 자세'},
    {'body_observations':{'torso':'forward'}},{'body_observations':{'knees':'straight'}},
    {'body_observations':{'knees':'mixed'}},{'search_text':'not folded arms'},
])
def test_unsupported_or_conflicting_text_is_not_silently_simplified(changes):
    with pytest.raises(ValueError):compile_plan(description(**changes))


def test_two_bone_exact_and_unreachable_does_not_stretch():
    start=np.array([12.,34.,-4.]);end=start+[.7,-.8,.2]
    middle=two_bone(start,end,.75,.65,[1,1,0])
    assert np.linalg.norm(middle-start)==pytest.approx(.75)
    assert np.linalg.norm(end-middle)==pytest.approx(.65)
    with pytest.raises(ValueError):two_bone(start,start+[4,0,0],.75,.65,[0,1,0])


def test_generator_does_not_reuse_input_motion_and_export_reloads(tmp_path):
    j,m=rig(order='ZYX');a=tmp_path/'a.bvh';b=tmp_path/'b.bvh'
    a.write_text(serialize_rig(j,m));b.write_text(serialize_rig(j,np.full_like(m,45)))
    saved=a.read_bytes();ra=TextConstraintPoseGenerator(a,max_variants=1).generate(description())
    rb=TextConstraintPoseGenerator(b,max_variants=1).generate(description())
    assert ra.status==rb.status=='preview_ready'
    assert ra.candidates[0]['bvh']==rb.candidates[0]['bvh']
    assert ra.candidates[0]['rig_source_sha256']!=rb.candidates[0]['rig_source_sha256']
    assert a.read_bytes()==saved
    artifact=tmp_path/'generated.bvh';artifact.write_text(ra.candidates[0]['bvh']);gj,gf=parse_bvh(str(artifact))
    assert validate_generated(gj,gf[0],compile_plan(description()))['kinematic_probe_pass']
    assert ra.to_dict()['release_allowed'] is ra.candidates[0]['refine_allowed'] is False
    assert ra.candidates[0]['library_lookup_supported'] is False
    json.dumps(ra.to_dict(),allow_nan=False)


def test_bad_rig_and_budget_errors_have_no_candidate(tmp_path):
    missing=TextConstraintPoseGenerator(tmp_path/'missing.bvh').generate(description())
    assert missing.status=='error' and not missing.candidates
    j,f=rig();path=tmp_path/'rig.bvh';path.write_text(serialize_rig(j,f))
    result=TextConstraintPoseGenerator(path,budget_ms=.000001).generate(description())
    assert result.status=='timeout' and not result.candidates
    for kwargs in ({'max_variants':4},{'budget_ms':float('nan')},{'max_variants':True}):
        with pytest.raises(ValueError):TextConstraintPoseGenerator(path,**kwargs)


class SpyGenerator:
    def __init__(self):self.calls=[]
    def generate(self,desc):
        self.calls.append(desc);return GenerationResult('unsupported','fixture')


@pytest.mark.parametrize('status,count,block',[('ok',1,False),('ok',0,False),('error',0,False),('no_result',1,False),('no_result',0,True)])
def test_fallback_only_for_successful_unblocked_empty_search(status,count,block):
    g=SpyGenerator();r=after_no_result(g,description(),search_status=status,hit_count=count,policy_block=block)
    assert r.status=='not_requested' and not g.calls


def test_empty_search_calls_generator_once_and_catches_broken_adapter():
    g=SpyGenerator();r=after_no_result(g,description(),search_status='no_result',hit_count=0)
    assert r.status=='unsupported' and len(g.calls)==1
    class Broken:
        def generate(self,d):raise RuntimeError('must not leak detail')
    r=after_no_result(Broken(),description(),search_status='no_result',hit_count=0)
    assert r.status=='error' and r.reason=='generation_exception:RuntimeError'


class VLM(MockVLMClient):
    def describe_people(self,image,boxes):
        return DescribeOutcome('ok',descriptions={i:description() for i in range(len(boxes))})


def run_pipeline(*,on=True,status='no_result',generator=None,hint='full_half standing front rough'):
    old=CFG.search_routing_enabled;CFG.search_routing_enabled=on
    try:
        pipeline=Pipeline(build_synthetic_index(),vlm_client=VLM(),pose_model=_RoughPose(),
                          semantic_searcher=_StubSearcher(status),pose_generator=generator)
        return pipeline.process_cut(_Img(hint))
    finally:CFG.search_routing_enabled=old


def test_pipeline_hook_disabled_by_default_and_search_hit_error_normal_never_generate():
    result=run_pipeline();assert 'generation' not in result.person_routing[0]
    for kwargs in ({'status':'ok'},{'status':'error'},{'on':False},{'hint':'full_half standing front'}):
        g=SpyGenerator();run_pipeline(generator=g,**kwargs);assert not g.calls


def test_pipeline_empty_search_generation_stays_separate_and_refine_blocked():
    g=SpyGenerator();before=run_pipeline();after=run_pipeline(generator=g)
    assert len(g.calls)==1
    assert _signature(before)==_signature(after)
    r=after.person_routing[0]
    assert r['route']=='semantic' and r['search']['status']=='no_result'
    assert r['generation']['status']=='unsupported' and not r['generation']['release_allowed']
    assert after.person_candidates==[[]] and after.person_semantic_candidates==[[]]
    assert not after.descriptors[0].refine_allowed


def test_explicit_cropped_scope_does_not_generate_full_body():
    g=SpyGenerator();result=run_pipeline(generator=g,hint='bust sitting front rough')
    assert not g.calls
