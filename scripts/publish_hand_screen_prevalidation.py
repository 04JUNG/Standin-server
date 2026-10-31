"""Publish completed staging checkpoints and verify startup installation in AWS.

Only derived certificates are published. Source BVHs/FBXs remain in AWS. This
does not set runtime feature flags or update an ECS service.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.launch_hand_screen_prevalidation import BUCKET, CLUSTER, FILES, code_package


def launch(receipt_path, output_path):
    import boto3
    from botocore.config import Config
    receipt = json.loads(receipt_path.read_text())
    session = boto3.Session(region_name='ap-northeast-2')
    if session.client('sts').get_caller_identity()['Account'] != '136889124221':
        raise ValueError('wrong staging account')
    ecs = session.client('ecs');s3 = session.client('s3',config=Config(signature_version='s3v4'))
    for job in receipt['jobs']:
        s3.head_object(Bucket=BUCKET,Key=job['summary_key'])
    retries=receipt.get('retries',[])
    for job in retries:
        s3.head_object(Bucket=BUCKET,Key=job['summary_key'])
    code=code_package();code_sha=hashlib.sha256(code).hexdigest()
    code_key=receipt['prefix']+'/publish-code-'+code_sha[:12]+'.tar.gz'
    s3.put_object(Bucket=BUCKET,Key=code_key,Body=code,IfNoneMatch='*',ContentType='application/gzip')
    folder = 'hand-screen/releases/'+receipt['version']
    def delegated(key):
        return dict(key=key,url=s3.generate_presigned_url('put_object',Params=dict(
            Bucket=BUCKET,Key=key,ContentType='application/json',IfNoneMatch='*'),ExpiresIn=3600))
    plan = dict(bucket=BUCKET,pose_count=receipt['library']['pose_count'],
        characters=receipt['characters'],hand_settings=receipt['hand_settings'],
        summaries=[j['summary_key'] for j in receipt['jobs']],
        retry_summaries=[j['summary_key'] for j in retries],
        catalog=delegated(folder+'/catalog.json'),manifest=delegated(folder+'/manifest.json'))
    raw=json.dumps(plan).encode();pin=hashlib.sha256(raw).hexdigest()
    plan_key=receipt['prefix']+'/private-publish-plan.json'
    s3.put_object(Bucket=BUCKET,Key=plan_key,Body=raw,IfNoneMatch='*',ContentType='application/json')
    # This is a verification-only task. Do not import/start inference or expose
    # ports, and do not copy the Converter API into the inference distribution.
    source=f'''import hashlib,io,json,os,runpy,sys,tarfile
from pathlib import Path
import boto3
s3=boto3.client('s3')
raw=s3.get_object(Bucket={BUCKET!r},Key={code_key!r})['Body'].read(1024*1024)
assert hashlib.sha256(raw).hexdigest()=={code_sha!r},'code checksum mismatch'
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
    members=tar.getmembers()
    assert len(members)=={len(FILES)} and {{m.name for m in members}}==set({FILES!r})
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk()
        if member.name=='converter_api/app.py':continue
        path=Path('/app')/member.name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(tar.extractfile(member).read())
raw=s3.get_object(Bucket={BUCKET!r},Key={plan_key!r})['Body'].read(1024*1024)
assert hashlib.sha256(raw).hexdigest()=={pin!r},'publish plan checksum mismatch'
Path('/tmp/hand-publish-plan.json').write_bytes(raw)
os.setgid(10001);os.setuid(10001);sys.path.insert(0,'/app')
sys.argv=['assemble_hand_screen_release.py','--plan','/tmp/hand-publish-plan.json']
runpy.run_path('/app/scripts/assemble_hand_screen_release.py',run_name='__main__')
from scripts.install_hand_screen_release import install
raw=s3.get_object(Bucket={BUCKET!r},Key={folder+'/manifest.json'!r})['Body'].read(65537)
manifest_sha=hashlib.sha256(raw).hexdigest()
def fetch(key,limit):
    response=s3.get_object(Bucket={BUCKET!r},Key=key)
    with response['Body'] as body:return body.read(limit+1)
installed=install(fetch,{{'HAND_SCREEN_RELEASE_MANIFEST_KEY':{folder+'/manifest.json'!r},'HAND_SCREEN_RELEASE_MANIFEST_SHA256':manifest_sha,'HAND_SCREEN_RELEASE_ROOT':'/tmp/verified-hand-screen-release'}})
assert Path('/tmp/verified-hand-screen-release/catalog.json').stat().st_mode & 0o222==0
print(json.dumps({{'event':'hand_screen_startup_install_verified','manifest_key':{folder+'/manifest.json'!r},'manifest_sha256':manifest_sha,'entries':installed['entries'],'catalog_sha256':installed['catalog_sha256'],'runtime_enabled':False}}),flush=True)
'''
    service=ecs.describe_services(cluster=CLUSTER,services=[
        'StandinStagingApp-InferenceService1C7A7625-H1ZZzbPm93oz'])['services'][0]
    base=ecs.describe_task_definition(taskDefinition=service['taskDefinition'])['taskDefinition']
    allowed={'taskRoleArn','executionRoleArn','networkMode','requiresCompatibilities',
             'runtimePlatform','ephemeralStorage','cpu','memory'}
    task={k:copy.deepcopy(v) for k,v in base.items() if k in allowed}
    worker=copy.deepcopy(next(c for c in base['containerDefinitions'] if c['name']=='inference'))
    for name in ('secrets','dependsOn','mountPoints','healthCheck','portMappings'):worker.pop(name,None)
    worker.update(user='0',command=['python','-u','-c',source])
    worker['logConfiguration']['options']['awslogs-stream-prefix']='hand-screen-publish'
    task.update(family='standin-staging-hand-screen-publish',containerDefinitions=[worker])
    definition=ecs.register_task_definition(**task)['taskDefinition']['taskDefinitionArn']
    result=ecs.run_task(cluster=CLUSTER,taskDefinition=definition,launchType='FARGATE',
        networkConfiguration=service['networkConfiguration'],count=1,startedBy='hand-screen-publish')
    if result.get('failures'):raise ValueError('publish job launch failed')
    output=dict(task=result['tasks'][0]['taskArn'],definition=definition,
        manifest_key=folder+'/manifest.json',catalog_key=folder+'/catalog.json',private_plan_key=plan_key)
    output_path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(event='hand_screen_publish_job_launched',**output)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output already exists')
    launch(args.receipt,args.output)
