from copy import deepcopy
import hashlib
import io
import json
import tarfile

import pytest

from scripts.install_hand_screen_release import install
from scripts.prepare_hand_screen_release_deployment import prepare, ROOT, INIT
from scripts.prevalidate_hand_screen_batch import fixtures
from scripts.prevalidate_hand_screen_batch import inspect_one
from scripts.assemble_hand_screen_release import assemble
from scripts.assemble_hand_screen_release import existing_manifest
from converter.hand_screen_catalog import binding
from tests.test_hand_skin_parallel import receipt
from tests.test_hand_screen_catalog import catalog_file


def release(tmp_path):
    path, pin, _ = catalog_file(tmp_path)
    raw = path.read_bytes()
    value = dict(schema='hand-screen-release.v1', preparation_complete=True,
        library_sha256='d'*64, profile_sha256='c'*64,
        catalog=dict(path='catalog.json', size=len(raw), sha256=pin, entries=3))
    blob = json.dumps(value).encode()
    key = 'hand-screen/releases/test-001/manifest.json'
    env = dict(HAND_SCREEN_RELEASE_MANIFEST_KEY=key,
        HAND_SCREEN_RELEASE_MANIFEST_SHA256=hashlib.sha256(blob).hexdigest(),
        HAND_SCREEN_RELEASE_ROOT=str(tmp_path/'installed'))
    objects = {key: blob, key.replace('manifest.json', 'catalog.json'): raw}
    return objects, env, value


def task():
    return dict(family='StandinStagingAppInferenceTaskExample',cpu='1024',memory='6144',
        taskRoleArn='preserved-role',volumes=[dict(name='body-assets',host={})],
        containerDefinitions=[dict(name='inference',image='old-image',
            environment=[dict(name='HAND_SKIN_SCREEN_CONCURRENCY',value='2')],
            secrets=[dict(name='GEMINI_API_KEY',valueFrom='preserved-reference')],
            dependsOn=[dict(containerName='body-assets-init',condition='SUCCESS')],
            mountPoints=[dict(sourceVolume='body-assets',containerPath='/app/body-staging',readOnly=True)]),
            dict(name='body-assets-init',image='unchanged-initializer')])


def test_verified_release_installs_read_only(tmp_path):
    objects,env,value=release(tmp_path)
    receipt=install(lambda key,limit:objects[key],env)
    output=tmp_path/'installed/catalog.json'
    assert output.read_bytes()==objects[next(k for k in objects if k.endswith('catalog.json'))]
    assert output.stat().st_mode & 0o222 == 0
    assert receipt['entries']==3 and receipt['catalog_sha256']==value['catalog']['sha256']


@pytest.mark.parametrize('problem',['manifest_hash','catalog_hash','profile','count','incomplete','traversal','duplicate'])
def test_bad_release_never_publishes_a_catalog(tmp_path,problem):
    objects,env,value=release(tmp_path)
    key=env['HAND_SCREEN_RELEASE_MANIFEST_KEY']
    if problem=='catalog_hash': objects[key.replace('manifest.json','catalog.json')]+=b' '
    elif problem=='profile':value['profile_sha256']='e'*64
    elif problem=='count':value['catalog']['entries']=2
    elif problem=='incomplete':value['preparation_complete']=False
    elif problem=='traversal':value['catalog']['path']='../catalog.json'
    objects[key]=json.dumps(value).encode()
    if problem=='duplicate':objects[key]=objects[key].replace(b'"schema":',b'"schema":"duplicate","schema":',1)
    if problem!='manifest_hash':env['HAND_SCREEN_RELEASE_MANIFEST_SHA256']=hashlib.sha256(objects[key]).hexdigest()
    else:env['HAND_SCREEN_RELEASE_MANIFEST_SHA256']='0'*64
    with pytest.raises(ValueError):install(lambda k,limit:objects[k],env)
    assert not (tmp_path/'installed/catalog.json').exists()


@pytest.mark.parametrize('enabled',['0','1'])
def test_task_waits_for_initializer_and_runtime_mount_is_read_only(tmp_path,enabled):
    objects,env,value=release(tmp_path);before=task();original=deepcopy(before)
    after=prepare(before,key=env['HAND_SCREEN_RELEASE_MANIFEST_KEY'],
        manifest_sha256=env['HAND_SCREEN_RELEASE_MANIFEST_SHA256'],
        manifest_raw=objects[env['HAND_SCREEN_RELEASE_MANIFEST_KEY']],
        image='136889124221.dkr.ecr.ap-northeast-2.amazonaws.com/standin/inference:'+'a'*40,enabled=enabled)
    assert before==original
    assert after['cpu']==before['cpu'] and after['memory']==before['memory']
    target=after['containerDefinitions'][0]
    assert target['image']==before['containerDefinitions'][0]['image']
    assert target['secrets']==before['containerDefinitions'][0]['secrets']
    assert dict(containerName=INIT,condition='SUCCESS') in target['dependsOn']
    assert dict(sourceVolume='hand-screen-release',containerPath=ROOT,readOnly=True) in target['mountPoints']
    settings={r['name']:r['value'] for r in target['environment']}
    assert settings['HAND_SKIN_SCREEN_CONCURRENCY']=='2'
    assert ('HAND_SKIN_PREVALIDATED_PATH' in settings)==(enabled=='1')
    assert after['containerDefinitions'][1]==before['containerDefinitions'][1]
    rolled_back=prepare(after,enabled='0')
    assert rolled_back==before


def test_empty_settings_preserve_task_exactly():
    assert prepare(task())==task()


def test_production_task_cannot_be_enabled(tmp_path):
    value=task();value['family']='StandinAppInferenceTaskProduction'
    with pytest.raises(ValueError):prepare(value,enabled='1')


def test_streamed_extraction_pins_archive_and_does_not_extract_thumbnails(tmp_path):
    path=tmp_path/'library.tar.gz'
    with tarfile.open(path,'w:gz') as tar:
        for name,data in [('bvh/a pose (1).bvh',b'BVH a'),('bvh/포즈.bvh',b'BVH b'),('thumbs/unrelated.png',b'private thumbnail')]:
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    pin=hashlib.sha256(path.read_bytes()).hexdigest();out=tmp_path/'extracted';out.mkdir()
    rows=fixtures(path,pin,2,0,1,out)
    assert [r['id'] for r in rows]==['a pose (1)','포즈']
    assert sorted(p.name for p in out.iterdir())==['a pose (1).bvh','포즈.bvh']
    with pytest.raises(ValueError,match='archive hash'):fixtures(path,'0'*64,2,0,1,out)


@pytest.mark.parametrize('status',['clear_screen','intersection'])
@pytest.mark.parametrize('budget',[14.85,60.])
def test_offline_inspection_preserves_valid_receipts(tmp_path,status,budget):
    path=tmp_path/'a.bvh';path.write_text('BVH a')
    identity=binding('a',hashlib.sha256(path.read_bytes()).hexdigest(),'master','a'*64,None,'c'*64)
    def endpoint(**kwargs):
        from converter_api.work_queue import current_work
        assert 0 < current_work.get().remaining <= budget
        return receipt(kwargs['bvh'].file.read().decode(),None,status)
    result=inspect_one(endpoint,identity,path,budget_seconds=budget)
    assert result['receipt']['status']==status and result['binding']==identity


def test_transient_inspection_failure_is_not_a_certificate(tmp_path):
    from converter_api.work_queue import current_work
    path=tmp_path/'a.bvh';path.write_text('BVH a')
    identity=binding('a',hashlib.sha256(path.read_bytes()).hexdigest(),'master','a'*64,None,'c'*64)
    calls=[]
    def endpoint(**kwargs):
        calls.append(1)
        raise TimeoutError('private diagnostic')
    result=inspect_one(endpoint,identity,path,attempts=2)
    assert len(calls)==2 and 'receipt' not in result
    assert result['unavailable']['reason']=='TimeoutError'
    assert current_work.get() is None


@pytest.mark.parametrize('problem',[None,'pilot','incomplete_shards','tamper','unavailable','body','count'])
def test_release_assembler_requires_complete_bound_checkpoints(tmp_path,problem):
    path,_,_=catalog_file(tmp_path)
    value=dict(schema='hand-screen-preparation-part.v1',shard=0,part=0,
        profile_sha256='c'*64,library_sha256='d'*64,catalog=json.loads(path.read_bytes()),unavailable=[])
    if problem=='unavailable':
        key=next(iter(value['catalog']['entries']))
        missing=value['catalog']['entries'].pop(key)
        value['unavailable']=[dict(binding=missing['binding'],reason='TimeoutError')]
    raw=json.dumps(value).encode()
    summary=dict(schema='hand-screen-preparation-job.v1',shard=0,shards=1,
        profile_sha256='c'*64,library_sha256='d'*64,pilot=False,
        counts=dict(attempted=3,clear_screen=2,intersection=1),
        hand_settings=[dict(id='original',hands=None)],
        target_profiles={'master':dict(character_sha256='a'*64)},
        parts=[dict(key='part.json',size=len(raw),sha256=hashlib.sha256(raw).hexdigest())])
    if problem=='pilot':summary['pilot']=True
    if problem=='incomplete_shards':summary['shards']=2
    if problem=='tamper':raw+=b' '
    if problem=='body':summary['target_profiles']['master']['character_sha256']='e'*64
    if problem=='count':summary['counts']['attempted']=2
    if problem=='unavailable':summary['counts']=dict(attempted=3,clear_screen=1,intersection=1,unavailable=1)
    def run():return assemble([summary],lambda key,size:raw,expected_pose_count=3,
        expected_characters=[dict(id='master',sha256='a'*64)],expected_hands=[dict(id='original',hands=None)])
    if problem:
        with pytest.raises(ValueError):run()
    else:
        catalog,manifest=run()
        assert manifest['catalog']['entries']==3 and manifest['preparation_complete']
        assert hashlib.sha256(catalog).hexdigest()==manifest['catalog']['sha256']


@pytest.mark.parametrize('problem',[None,'unmatched','duplicate','still_unavailable'])
def test_retry_certificate_must_match_original_failed_binding(tmp_path,problem):
    from converter.hand_screen_catalog import make_catalog, digest
    path,_,_=catalog_file(tmp_path)
    original=json.loads(path.read_bytes())
    key=next(k for k,r in original['entries'].items() if r['receipt']['status']=='clear_screen')
    row=original['entries'].pop(key)
    failed=deepcopy(row['binding'])
    if problem=='unmatched':
        row=deepcopy(row)
        row['binding']['pose_id']='different'
    retry_catalog=make_catalog([(row['binding'],row['receipt'])])
    source=dict(schema='hand-screen-preparation-part.v1',shard=0,part=0,
        profile_sha256='c'*64,library_sha256='d'*64,catalog=original,
        unavailable=[dict(binding=failed,reason='WORK_BUDGET_EXHAUSTED')])
    retry=dict(source,catalog=retry_catalog,unavailable=[])
    if problem=='still_unavailable':
        retry.update(catalog=None,unavailable=[dict(binding=failed,reason='WORK_BUDGET_EXHAUSTED')])
    blobs={'source':json.dumps(source).encode(),'retry':json.dumps(retry).encode()}
    def part(name):return dict(key=name,size=len(blobs[name]),sha256=hashlib.sha256(blobs[name]).hexdigest())
    base=dict(profile_sha256='c'*64,library_sha256='d'*64,
        hand_settings=[dict(id='original',hands=None)],
        target_profiles={'master':dict(character_sha256='a'*64)})
    summary=dict(base,schema='hand-screen-preparation-job.v1',shard=0,shards=1,pilot=False,
        counts=dict(attempted=3,clear_screen=1,intersection=1,unavailable=1),parts=[part('source')])
    repaired=dict(base,schema='hand-screen-preparation-retry.v1',
        counts=dict(attempted=1,**{'unavailable' if problem=='still_unavailable' else 'clear_screen':1}),parts=[part('retry')])
    def run():return assemble([summary],lambda key,size:blobs[key],expected_pose_count=3,
        expected_characters=[dict(id='master',sha256='a'*64)],expected_hands=base['hand_settings'],
        retry_summaries=[repaired,repaired] if problem=='duplicate' else [repaired])
    if problem:
        with pytest.raises(ValueError):run()
    else:
        catalog,manifest=run()
        assert len(json.loads(catalog)['entries'])==3 and manifest['reinspected_entries']==1


def test_publishing_retry_keeps_first_timestamp_but_rejects_changed_contents():
    prior=dict(created_at='first',profile_sha256='c'*64,catalog=dict(sha256='a'*64))
    fresh=dict(prior,created_at='later')
    assert existing_manifest(fresh,prior)==prior
    with pytest.raises(ValueError):existing_manifest(dict(fresh,profile_sha256='d'*64),prior)
    with pytest.raises(ValueError):existing_manifest(dict(fresh,catalog=dict(sha256='b'*64)),prior)
