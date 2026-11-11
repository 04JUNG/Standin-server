"""Launch bounded offline jobs in the existing staging AWS account only.

No service update, production target, source asset download, or IAM expansion.
Each job gets immutable output URLs rather than general S3 write permission.
"""
import argparse
import copy
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
BUCKET = 'standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk'
CLUSTER = 'StandinStagingApp-ClusterEB0386A7-fTa8F6KCe561'
SERVICE = 'StandinStagingApp-ConverterServiceDE3E8133-81qvalM5m34c'
BASE_IMAGE = '136889124221.dkr.ecr.ap-northeast-2.amazonaws.com/standin/converter@sha256:b60b78bffadab4cc07ed71ef33cd8dcecd3a1eac3e04c846fe57a24a27782898'
BASE_API_SHA = '2613114ebee7167cc564917b6ea91fc03ac8248443fd48c93a16d5f48e6961df'
FILES = ['converter/hand_screen_catalog.py', 'converter_api/app.py',
         'scripts/prevalidate_hand_screen_batch.py', 'scripts/assemble_hand_screen_release.py',
         'scripts/install_hand_screen_release.py', 'scripts/retry_hand_screen_prevalidation.py']
CHARACTERS = [dict(id='standin-master-v2',sha256='7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d'),
              dict(id='standin-female-v2-lbs',sha256='96cd01717ed48ce416333c21dc68e8d62d185e91e016056c9eae305e8c37a186')]
LIBRARY = dict(bucket=BUCKET,
    key='pose-library/releases/lib-20261021-21b1b3d9-4scopes-8views-297e8f1ea240.tar.gz',
    sha256='297e8f1ea240961964abd722704b92a0f20bfe190ece400eec156a568f1c0a80',pose_count=5862)
HANDS = [dict(id='both-relaxed',hands=dict(left='relaxed',right='relaxed'))]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code_package():
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as tar:
            for name in FILES:
                raw = (ROOT/name).read_bytes()
                info = tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.mtime=0
                tar.addfile(info,io.BytesIO(raw))
    return buffer.getvalue()


def bootstrap(code_key, code_sha, plan_key, plan_sha, script='prevalidate_hand_screen_batch.py'):
    # Install proposed profile bytes only in the isolated preparation job. They
    # match the pinned image proposed by PR #163; this is not a serving rollout.
    return f'''import hashlib,io,os,runpy,sys,tarfile
from pathlib import Path
import boto3
s3=boto3.client('s3')
raw=s3.get_object(Bucket={BUCKET!r},Key={code_key!r})['Body'].read(1024*1024)
assert hashlib.sha256(raw).hexdigest()=={code_sha!r},'preparation code hash mismatch'
assert hashlib.sha256(Path('/app/converter_api/app.py').read_bytes()).hexdigest()=={BASE_API_SHA!r},'unexpected base API'
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
    members=tar.getmembers()
    assert len(members)=={len(FILES)} and {{m.name for m in members}}==set({FILES!r})
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk()
        path=Path('/app')/member.name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(tar.extractfile(member).read())
raw=s3.get_object(Bucket={BUCKET!r},Key={plan_key!r})['Body'].read(1024*1024)
assert hashlib.sha256(raw).hexdigest()=={plan_sha!r},'preparation plan hash mismatch'
Path('/tmp/hand-screen-plan.json').write_bytes(raw)
os.setgid(10001);os.setuid(10001)
sys.path.insert(0,'/app')
sys.argv=[{script!r},'--plan','/tmp/hand-screen-plan.json']
runpy.run_path('/app/scripts/'+{script!r},run_name='__main__')
'''


def launch(shards, pilot, output, inspection_budget_seconds=14.85, inspection_attempts=2):
    import boto3
    from botocore.config import Config
    if (not 1 <= shards <= 8 or (pilot and shards != 1)
            or not 1 <= inspection_budget_seconds <= 120 or not 1 <= inspection_attempts <= 3):
        raise ValueError('at most eight bounded offline jobs; pilot uses one')
    session = boto3.Session(region_name='ap-northeast-2')
    if session.client('sts').get_caller_identity()['Account'] != '136889124221':
        raise ValueError('wrong staging account')
    ecs = session.client('ecs')
    s3 = session.client('s3',config=Config(signature_version='s3v4'))
    service = ecs.describe_services(cluster=CLUSTER,services=[SERVICE])['services'][0]
    base = ecs.describe_task_definition(taskDefinition=service['taskDefinition'])['taskDefinition']
    actual = next(c for c in base['containerDefinitions'] if c['name']=='converter')
    if actual['image'] != BASE_IMAGE:
        raise ValueError('staging image changed; review target distribution first')
    raw = code_package();code_sha = sha(raw)
    version = 'prevalidation-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d')+'-'+code_sha[:8]+'-'+uuid.uuid4().hex[:6]
    prefix = 'hand-screen/preparation/'+version
    code_key = prefix+'/code.tar.gz'
    s3.put_object(Bucket=BUCKET,Key=code_key,Body=raw,IfNoneMatch='*',ContentType='application/gzip')
    def delegated(key):
        return dict(key=key,url=s3.generate_presigned_url('put_object',Params=dict(
            Bucket=BUCKET,Key=key,ContentType='application/json',IfNoneMatch='*'),ExpiresIn=21600))
    receipt = dict(version=version,prefix=prefix,code_key=code_key,code_sha256=code_sha,
                   library=LIBRARY,characters=CHARACTERS,hand_settings=HANDS,jobs=[])
    for shard in range(shards):
        folder = prefix+'/shard-'+str(shard).zfill(3)
        plan = dict(schema='hand-screen-preparation-plan.v1',shard=shard,shards=shards,
            workers=4,checkpoint_every=128,max_seconds=7200,
            inspection_budget_seconds=inspection_budget_seconds,inspection_attempts=inspection_attempts,
            library=LIBRARY,characters=CHARACTERS,hand_settings=HANDS,
            output_bucket=BUCKET,
            parts=[delegated(folder+'/part-'+str(i).zfill(3)+'.json') for i in range(92)],
            summary=delegated(folder+'/summary.json'))
        if pilot:plan['limit']=8
        blob = json.dumps(plan).encode();plan_key=folder+'/private-job-plan.json'
        s3.put_object(Bucket=BUCKET,Key=plan_key,Body=blob,IfNoneMatch='*',ContentType='application/json')
        allowed = {'taskRoleArn','executionRoleArn','networkMode','containerDefinitions',
                   'volumes','requiresCompatibilities','runtimePlatform','ephemeralStorage'}
        task = {k:copy.deepcopy(v) for k,v in base.items() if k in allowed}
        task.update(family='standin-staging-hand-prevalidation',cpu='8192',memory='16384')
        worker = next(c for c in task['containerDefinitions'] if c['name']=='converter')
        for key in ('healthCheck','portMappings','secrets'):worker.pop(key,None)
        worker['user']='0'
        worker['command']=['python','-u','-c',bootstrap(code_key,code_sha,plan_key,sha(blob))]
        env={r['name']:r['value'] for r in worker.get('environment',[])}
        env.update(CONVERTER_MAX_CONCURRENT_PROCESSES='4',CONVERTER_LOG_LEVEL='ERROR')
        env['CONVERTER_TIMEOUT_SECONDS']=str(max(30.,inspection_budget_seconds))
        worker['environment']=[dict(name=k,value=v) for k,v in env.items()]
        worker['logConfiguration']['options']['awslogs-stream-prefix']='hand-prevalidation'
        definition=ecs.register_task_definition(**task)['taskDefinition']['taskDefinitionArn']
        result=ecs.run_task(cluster=CLUSTER,taskDefinition=definition,launchType='FARGATE',
            networkConfiguration=service['networkConfiguration'],count=1,
            startedBy='hand-screen-prevalidation')
        if result.get('failures'):raise ValueError('offline job launch failed')
        item=dict(shard=shard,task=result['tasks'][0]['taskArn'],definition=definition,
                  plan_key=plan_key,summary_key=folder+'/summary.json')
        receipt['jobs'].append(item)
        output.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(event='prevalidation_job_launched',**item)),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards',type=int,default=4)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--inspection-budget-seconds',type=float,default=14.85)
    parser.add_argument('--inspection-attempts',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output already exists')
    launch(args.shards,args.pilot,args.output,args.inspection_budget_seconds,args.inspection_attempts)
