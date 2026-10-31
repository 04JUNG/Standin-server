"""Launch one bounded AWS-only retry worker for fixed preparation failures."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.launch_hand_screen_prevalidation import (
    BUCKET,CLUSTER,SERVICE,BASE_IMAGE,bootstrap,code_package)


def launch(path):
    import boto3
    from botocore.config import Config
    receipt=json.loads(path.read_text())
    if receipt.get('retries'):raise ValueError('retry worker already registered')
    session=boto3.Session(region_name='ap-northeast-2')
    if session.client('sts').get_caller_identity()['Account']!='136889124221':
        raise ValueError('wrong staging account')
    ecs=session.client('ecs');s3=session.client('s3',config=Config(signature_version='s3v4'))
    service=ecs.describe_services(cluster=CLUSTER,services=[SERVICE])['services'][0]
    base=ecs.describe_task_definition(taskDefinition=service['taskDefinition'])['taskDefinition']
    if next(c for c in base['containerDefinitions'] if c['name']=='converter')['image']!=BASE_IMAGE:
        raise ValueError('staging base image changed')
    part=receipt['jobs'][0]['summary_key'].rsplit('/',1)[0]+'/part-000.json'
    sample=json.loads(s3.get_object(Bucket=BUCKET,Key=part)['Body'].read(4*1024*1024))
    profile=sample['profile_sha256']
    code=code_package();pin=hashlib.sha256(code).hexdigest();code_key=receipt['prefix']+'/retry-code-'+pin[:12]+'.tar.gz'
    s3.put_object(Bucket=BUCKET,Key=code_key,Body=code,IfNoneMatch='*',ContentType='application/gzip')
    folder=receipt['prefix']+'/retry'
    def delegated(key):
        return dict(key=key,url=s3.generate_presigned_url('put_object',Params=dict(
            Bucket=BUCKET,Key=key,ContentType='application/json',IfNoneMatch='*'),ExpiresIn=21600))
    plan=dict(library=receipt['library'],characters=receipt['characters'],
        hand_settings=receipt['hand_settings'],profile_sha256=profile,
        source_jobs=receipt['jobs'],output_bucket=BUCKET,max_seconds=10800,
        parts=[delegated(folder+'/part-'+str(i).zfill(3)+'.json') for i in range(512)],
        summary=delegated(folder+'/summary.json'))
    raw=json.dumps(plan).encode();plan_key=folder+'/private-job-plan.json'
    s3.put_object(Bucket=BUCKET,Key=plan_key,Body=raw,IfNoneMatch='*',ContentType='application/json')
    allowed={'taskRoleArn','executionRoleArn','networkMode','containerDefinitions','volumes',
             'requiresCompatibilities','runtimePlatform','ephemeralStorage'}
    task={k:copy.deepcopy(v) for k,v in base.items() if k in allowed}
    task.update(family='standin-staging-hand-prevalidation-retry',cpu='8192',memory='16384')
    worker=next(c for c in task['containerDefinitions'] if c['name']=='converter')
    for key in ('healthCheck','portMappings','secrets'):worker.pop(key,None)
    worker['user']='0';worker['command']=['python','-u','-c',bootstrap(
        code_key,pin,plan_key,hashlib.sha256(raw).hexdigest(),'retry_hand_screen_prevalidation.py')]
    env={r['name']:r['value'] for r in worker.get('environment',[])}
    env.update(CONVERTER_MAX_CONCURRENT_PROCESSES='1',CONVERTER_LOG_LEVEL='ERROR')
    worker['environment']=[dict(name=k,value=v) for k,v in env.items()]
    worker['logConfiguration']['options']['awslogs-stream-prefix']='hand-prevalidation-retry'
    definition=ecs.register_task_definition(**task)['taskDefinition']['taskDefinitionArn']
    result=ecs.run_task(cluster=CLUSTER,taskDefinition=definition,launchType='FARGATE',
        networkConfiguration=service['networkConfiguration'],count=1,startedBy='hand-screen-prevalidation-retry')
    if result.get('failures'):raise ValueError('retry job launch failed')
    item=dict(task=result['tasks'][0]['taskArn'],definition=definition,
              plan_key=plan_key,summary_key=folder+'/summary.json')
    receipt['retries']=[item]
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(event='prevalidation_retry_launched',**item)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    launch(parser.parse_args().receipt)
