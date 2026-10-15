from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from src.natural_parts import NaturalPartIndex,VERSION,digest,validate_part
from src.natural_part_caption import parse_caption,prompt_for
from src.natural_part_search import NaturalPartSearch


def part(text='관찰한 자세',group='arms_down',visible=True):
    return dict(group=group,confidence='medium' if visible else 'unknown',
        observations=[text] if visible else [],description=text if visible else '',visible=visible,searchable=visible)


class Encoder:
    name='test-encoder';model_sha256='test-model';tokenizer_sha256='test-tokenizer'
    def __init__(self):self.fit_calls=[];self.encode_calls=[]
    def embed(self,text):
        v=np.array([float('높이' in text),float('아래' in text),float('펴고' in text),float('앉아' in text),.01])
        return v/np.linalg.norm(v)
    def fit(self,docs):self.fit_calls.append(docs);return np.stack([self.embed(t)for t in docs])
    def encode(self,text):self.encode_calls.append(text);return np.stack([self.embed(text)])


@pytest.fixture
def corpus(tmp_path):
    bundle=tmp_path/'bundle';(bundle/'bvh').mkdir(parents=True)
    conn=sqlite3.connect(bundle/'poses.db');conn.execute('create table poses(pose_id text,meta_json text)')
    records=[]
    for pid,upper,lower,ug,lg in (
        ('a','손을 높이 든다','다리를 펴고 선다','arms_raised','standing'),
        ('a_mirror','손을 높이 든다','다리를 펴고 선다','arms_raised','standing'),
        ('b','손을 아래로 내린다','다리를 접고 앉아 있다','arms_down','chair_sitting')):
        path=bundle/'bvh'/(pid+'.bvh');path.write_text('fixture '+pid)
        sha=digest(path);conn.execute('insert into poses values (?,?)',(pid,json.dumps({'bvh_sha256':sha})))
        records.append(dict(pose_id=pid,bvh_sha256=sha,upper=part(upper,ug),lower=part(lower,lg)))
    conn.commit();conn.close()
    tags=tmp_path/'part-tags.json'
    tags.write_text(json.dumps(dict(version=VERSION,library_version='fixture',
        source_db_sha256=digest(bundle/'poses.db'),poses=records)))
    return bundle,tags


def test_parts_are_indexed_independently_and_no_group_hard_filter(corpus):
    encoder=Encoder();index=NaturalPartIndex(*corpus,encoder)
    upper=index.search(part('손을 높이 든다','arms_down'),'upper')
    lower=index.search(part('다리를 접고 앉아 있다','standing'),'lower')
    assert upper['hits'][0]['pose_id']=='a' and lower['hits'][0]['pose_id']=='b'
    assert len(upper['hits'])==2 and len({h['family']for h in upper['hits']})==2
    assert not index.identity['group_hard_filter'] and not index.identity['skeleton_bonus']
    assert all(not h['release_allowed'] for h in upper['hits'])


def test_invisible_region_abstains_before_embedding(corpus):
    encoder=Encoder();index=NaturalPartIndex(*corpus,encoder)
    result=index.search(part(group='unknown',visible=False),'lower')
    assert result['status']=='abstained' and not result['hits'] and not encoder.encode_calls
    with pytest.raises(ValueError,match='abstain'):
        validate_part(dict(part('서 있다','standing'),visible=False),'lower',rough=True)


@pytest.mark.parametrize('field',('db','bvh','coverage','metadata_hash','version'))
def test_current_db_bvh_and_full_coverage_are_required(corpus,field):
    bundle,tags=corpus;data=json.loads(tags.read_text())
    if field=='db':data['source_db_sha256']='old'
    if field=='bvh':(bundle/'bvh/a.bvh').write_text('tampered')
    if field=='coverage':data['poses'].pop()
    if field=='version':data['version']='old-schema'
    if field=='metadata_hash':data['poses'][0]['bvh_sha256']='fake'
    tags.write_text(json.dumps(data))
    with pytest.raises(ValueError):NaturalPartIndex(bundle,tags,Encoder())


def test_quarantine_and_empty_corpus_fail_closed(corpus):
    index=NaturalPartIndex(*corpus,Encoder(),excluded=('a','a_mirror'))
    assert index.identity['eligible']==1 and index.search(part('손을 높이 든다'),'upper')['hits'][0]['pose_id']=='b'
    with pytest.raises(ValueError,match='no eligible'):
        NaturalPartIndex(*corpus,Encoder(),excluded=('a','a_mirror','b'))


def test_caption_rejects_missing_duplicate_or_hallucinated_hidden_parts():
    row=dict(index=0,subject_kind='human',upper=part(),lower=part(group='unknown',visible=False))
    assert parse_caption(json.dumps({'people':[row]}),1)[0]['lower']['group']=='unknown'
    with pytest.raises(ValueError):parse_caption(json.dumps({'people':[row,row]}),2)
    with pytest.raises(ValueError):parse_caption(json.dumps({'people':[]}),1)
    row['lower']['description']='서 있다'
    with pytest.raises(ValueError):parse_caption(json.dumps({'people':[row]}),1)
    prompt=prompt_for([[0,0,1,1]])
    assert '좌표/관절 각도는 생성하지 않는다' in prompt


def minimal_bvh(path):
    path.write_text('HIERARCHY\nROOT Hips\n{\n OFFSET 0 0 0\n CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n End Site\n { OFFSET 0 1 0 }\n}\nMOTION\nFrames: 1\nFrame Time: 0.033333\n0 0 0 0 0 0\n')


class SameSourceIndex:
    def __init__(self,path):self.path=path
    def source(self,pid):return self.path
    def search(self,query,part,top_k=5):
        if not query['visible']:return dict(status='abstained',hits=[],reason='not_visible')
        return dict(status='ok',hits=[dict(pose_id='same',part=part,score=.8,
            tag={'group':'standing' if part=='lower' else 'arms_down','description':'기본 자세'})])


def test_same_pose_is_not_spliced_or_padded_and_returned_bytes_checked(tmp_path):
    path=tmp_path/'source.bvh';minimal_bvh(path);before=path.read_bytes()
    service=NaturalPartSearch(SameSourceIndex(path))
    query={'upper':part(),'lower':part('다리를 편다','standing')}
    result=service.search(query)
    assert len(result['selected'])==1 and result['selected'][0]['kind']=='library_reference'
    assert result['selected'][0]['roundtrip_delta_torso']<1e-6
    assert not result['selected'][0]['release_allowed'] and path.read_bytes()==before
    assert not result['skeleton_used_for_ranking'] and not result['generation_enabled']


def test_hidden_lower_is_explicit_inferred_default_and_both_hidden_no_result(tmp_path):
    path=tmp_path/'source.bvh';minimal_bvh(path)
    service=NaturalPartSearch(SameSourceIndex(path))
    query={'upper':part(),'lower':part(group='unknown',visible=False)}
    original=deepcopy(query);result=service.search(query)
    assert query==original and result['remove_lower_body'] and result['output_scope']=='half'
    assert result['lower']['status']=='abstained' and not result['lower']['hits']
    assert result['lower']['default_fill'][0]['inferred']
    assert result['selected'][0]['inferred_parts']==['lower']
    query['upper']=part(group='unknown',visible=False)
    assert service.search(query)['status']=='not_run'


def test_rejected_composition_cannot_leak_as_a_candidate(tmp_path,monkeypatch):
    path=tmp_path/'source.bvh';minimal_bvh(path)
    index=SameSourceIndex(path)
    def search(query,part,top_k=5):
        return dict(status='ok',hits=[dict(pose_id=part,part=part,score=.8,tag={'group':'standing'})])
    index.search=search
    monkeypatch.setattr('src.natural_part_search.compose_upper_lower',lambda *a:SimpleNamespace(
        frame=np.zeros(6),diagnostics={'pass':False,'reasons':['collision']}))
    result=NaturalPartSearch(index).search({'upper':part(),'lower':part(group='standing')})
    assert result['status']=='no_result' and result['selected']==[]
    assert result['attempts'][0]['status']=='rejected'


def test_nonhuman_and_visible_but_insufficient_details_do_not_force_pose_search(tmp_path):
    path=tmp_path/'source.bvh';minimal_bvh(path)
    service=NaturalPartSearch(SameSourceIndex(path))
    query={'subject_kind':'nonhuman','upper':part(),'lower':part(group='standing')}
    assert service.search(query)['status']=='policy_skip'
    query['subject_kind']='human'
    for p in ('upper','lower'):query[p]['searchable']=False
    assert service.search(query)['status']=='not_run'


def test_unobserved_default_lower_cannot_stand_up_a_prone_upper():
    from tests.test_natural_split_compose import _posed
    from src.natural_part_search import align_inferred_lower
    from src.bvh import fk,coco17_from_fk
    from src.experimental.split_compose import compose_upper_lower
    bj,bf=_posed();uj,uf=_posed(arm_left=70,arm_right=-70)
    uf[4]=90 # X root tilt: test rig rotation channels Z/X/Y.
    original=bf.copy();aligned,diag=align_inferred_lower(bj,bf,uj,uf)
    assert np.array_equal(bf,original) and diag['alignment_error']<1e-6
    result=compose_upper_lower(bj,aligned,uj,uf)
    actual,_=coco17_from_fk(bj,fk(bj,result.frame));donor,_=coco17_from_fk(uj,fk(uj,uf))
    for a,b in ((5,7),(7,9),(6,8),(8,10),(11,5)):
        va,vb=actual[b]-actual[a],donor[b]-donor[a]
        assert np.dot(va,vb)/np.linalg.norm(va)/np.linalg.norm(vb)>.9999


def write_rig(path,joints,frame):
    def node(i):
        name,parent,offset,channels,is_end=joints[i]
        start=('ROOT' if parent==-1 else 'JOINT')+' '+name
        children=[k for k,j in enumerate(joints)if j[1]==i]
        return start+'\n{\n OFFSET '+' '.join(map(str,offset))+'\n CHANNELS '+str(len(channels))+' '+' '.join(channels)+'\n'+''.join(node(k)for k in children)+'}\n'
    path.write_text('HIERARCHY\n'+node(0)+'MOTION\nFrames: 1\nFrame Time: 0.033333\n'+' '.join(map(str,frame))+'\n')


def test_actual_cross_source_composition_preserves_observed_lower_and_sources(tmp_path):
    from tests.test_natural_split_compose import _posed
    from src.bvh import parse_bvh,fk
    bj,bf=_posed(knee=80);uj,uf=_posed(arm_left=70,arm_right=-70)
    base=tmp_path/'lower.bvh';donor=tmp_path/'upper.bvh'
    write_rig(base,bj,bf);write_rig(donor,uj,uf)
    class Index:
        def source(self,pid):return donor if pid=='upper' else base
        def search(self,query,part,top_k=5):return dict(status='ok',hits=[dict(
            pose_id=part,score=.8,tag={'group':'standing' if part=='lower' else 'arms_raised','description':'test'})])
    before=(base.read_bytes(),donor.read_bytes())
    result=NaturalPartSearch(Index()).search({'upper':part(),'lower':part(group='standing')})
    assert len(result['selected'])==1,result['attempts']
    hit=result['selected'][0];assert hit['diagnostics']['pass'] and hit['kind']=='composed_preview'
    assert 'inferred_lower_alignment' not in hit['diagnostics']
    saved=tmp_path/'returned.bvh';saved.write_text(hit['bvh']);rj,rf=parse_bvh(str(saved))
    after=fk(rj,rf[0]);original=fk(bj,bf)
    for i,j in enumerate(bj):
        if 'Leg' in j[0] or 'Foot' in j[0] or j[0]=='Hips':np.testing.assert_allclose(after[i],original[i],atol=1e-6)
    assert before==(base.read_bytes(),donor.read_bytes())
