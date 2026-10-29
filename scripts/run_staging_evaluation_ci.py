#!/usr/bin/env python3
"""Launch one bounded evaluation task using the current staging service definition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standin_eval.rubric_package import _extract
from standin_eval.util import read_json, write_json

STAGING_CLUSTER = 'StandinStagingApp-ClusterEB0386A7-fTa8F6KCe561'
STAGING_SERVICE = 'StandinStagingApp-InferenceService1C7A7625-H1ZZzbPm93oz'
BUCKET = 'standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk'
DATA_SHA = 'be85f93b2e3f4bc276d221ce8dd5c716b3360f546802c5dcf65a703d7bcf00b3'
DATA_KEY = 'evaluation/fixed50/v3.2/' + DATA_SHA + '.zip'
PREFIX = 'evaluation/staging/fixed50-v32'


def bootstrap():
    # Only our generated, hashed code archive is extracted here (not the dataset).
    return "import os,urllib.request,hashlib,zipfile,sys; p='/tmp/eval-code.zip'; urllib.request.urlretrieve(os.environ['EVAL_CODE_URL'],p); assert hashlib.sha256(open(p,'rb').read()).hexdigest()==os.environ['EVAL_CODE_SHA256']; zipfile.ZipFile(p).extractall('/tmp/eval-code'); os.environ['PYTHONPATH']='/tmp/eval-code:/app'; os.execv(sys.executable,[sys.executable,'-c',\"import sys,runpy; sys.path.insert(0,'/tmp/eval-code'); runpy.run_module('standin_eval.staging_live',run_name='__main__')\"])"


def evaluation_definition(definition):
    allowed = ('taskRoleArn','executionRoleArn','networkMode','containerDefinitions','volumes',
               'placementConstraints','requiresCompatibilities','cpu','memory','runtimePlatform','ephemeralStorage')
    value = {k:definition[k] for k in allowed if k in definition}
    value = json.loads(json.dumps(value))
    value['family'] = 'standin-staging-fixed50-evaluation'
    current_image = next(c['image'] for c in value['containerDefinitions'] if c['name'] == 'inference')
    for container in value['containerDefinitions']:
        if container['name'] == 'body-assets-init':
            if container.get('command', [])[:2] != ['python', '-c']:
                raise ValueError('unknown asset initializer contract')
            # Initializer is inline stdlib+boto3 code, not code inside its old image.
            # Avoid a deleted historical digest while preserving its pinned manifest.
            container['image'] = current_image
        if container['name'] == 'inference':
            container['command'] = ['python', '-c', bootstrap()]
            container.pop('healthCheck', None)
            container['secrets'] = [s for s in container.get('secrets', []) if not s['name'].startswith('DISCORD_')]
            container['environment'] = [e for e in container.get('environment', []) if not e['name'].startswith('DISCORD_')]
    return value


def main():
    import boto3
    p = argparse.ArgumentParser()
    p.add_argument('--revision', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--judge-model', required=True)
    p.add_argument('--output', default='out/staging-evaluation')
    p.add_argument('--timeout-seconds', type=int, default=3000)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    if not args.judge_model.strip() or not 1 <= args.timeout_seconds <= 3000:
        raise ValueError('explicit judge model and deadline <= 3000 seconds required')
    if not args.run_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('invalid run ID')
    ecs, s3 = boto3.client('ecs', region_name='ap-northeast-2'), boto3.client('s3', region_name='ap-northeast-2')
    service = ecs.describe_services(cluster=STAGING_CLUSTER, services=[STAGING_SERVICE])['services'][0]
    definition = ecs.describe_task_definition(taskDefinition=service['taskDefinition'])['taskDefinition']
    inference = next(c for c in definition['containerDefinitions'] if c['name'] == 'inference')
    env = {e['name']:e['value'] for e in inference.get('environment', [])}
    if env.get('DEPLOYMENT_VERSION') != args.revision or not inference['image'].endswith(':'+args.revision):
        raise ValueError('staging service does not run requested immutable revision')
    head = s3.head_object(Bucket=BUCKET, Key=DATA_KEY)
    if head.get('Metadata', {}).get('sha256') != DATA_SHA:
        raise ValueError('pinned staging dataset not provisioned')
    if args.preflight_only:
        print(json.dumps({'staging_revision': args.revision, 'dataset_sha256': DATA_SHA, 'status': 'ready'})); return
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp)/'code.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for folder, pattern in [('standin_eval','*.py'), ('evaluation/rubrics','*.json')]:
                for source in sorted((ROOT/folder).rglob(pattern)):
                    info = zipfile.ZipInfo(source.relative_to(ROOT).as_posix(), (2026,1,1,0,0,0))
                    z.writestr(info, source.read_bytes())
        code_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        # Different evaluator code/model creates a separate series, never a misleading delta.
        contract = hashlib.sha256((code_hash+args.judge_model+DATA_SHA).encode()).hexdigest()
        series = PREFIX+'/'+contract
        run_key = series+'/runs/'+args.run_id
        s3.upload_file(str(archive), BUCKET, run_key+'/code.zip')
        def signed(method, key):
            return s3.generate_presigned_url(method, Params={'Bucket':BUCKET,'Key':key}, ExpiresIn=3300)
        previous_key = series+'/baseline.json'
        previous_url = ''
        existing = s3.list_objects_v2(Bucket=BUCKET, Prefix=previous_key, MaxKeys=1).get('Contents', [])
        if any(item['Key'] == previous_key for item in existing):
            previous = s3.get_object(Bucket=BUCKET, Key=previous_key)['Body'].read()
            # Snapshot previous baseline: never depend on a mutable URL during execution.
            s3.put_object(Bucket=BUCKET, Key=run_key+'/previous.json', Body=previous)
            previous_url = signed('get_object', run_key+'/previous.json')
        values = {'EVAL_CODE_URL':signed('get_object',run_key+'/code.zip'), 'EVAL_CODE_SHA256':code_hash,
            'EVAL_DATA_URL':signed('get_object',DATA_KEY), 'EVAL_DATA_SHA256':DATA_SHA,
            'EVAL_RESULT_URL':signed('put_object',run_key+'/report.zip'), 'EVAL_PREVIOUS_URL':previous_url,
            'EVAL_REVISION':args.revision,'EVAL_JUDGE_MODEL':args.judge_model,'EVAL_CONTRACT_VERSION':contract}
        registered = ecs.register_task_definition(**evaluation_definition(definition))['taskDefinition']['taskDefinitionArn']
        task_arn = None
        try:
            response = ecs.run_task(cluster=STAGING_CLUSTER, taskDefinition=registered, launchType='FARGATE',
                networkConfiguration=service['networkConfiguration'], count=1, startedBy='fixed50-staging-eval',
                tags=[{'key':'StandinEvaluation','value':'fixed50'}],
                overrides={'containerOverrides':[{'name':'inference','environment':[{'name':k,'value':v} for k,v in values.items()]}]})
            if response.get('failures') or not response.get('tasks'):
                raise RuntimeError('evaluation task launch failed')
            task_arn = response['tasks'][0]['taskArn']
            write_json(output/'task.json', {'task':task_arn,'revision':args.revision,'dataset':DATA_SHA,'contract':contract})
            deadline = time.monotonic()+args.timeout_seconds
            while time.monotonic() < deadline:
                task = ecs.describe_tasks(cluster=STAGING_CLUSTER,tasks=[task_arn])['tasks'][0]
                if task['lastStatus'] == 'STOPPED':
                    break
                print(json.dumps({'status':task['lastStatus'],'elapsed_seconds':int(args.timeout_seconds-(deadline-time.monotonic()))}),flush=True)
                time.sleep(20)
            else:
                raise TimeoutError('staging evaluation deadline exceeded')
            write_json(output/'task-stop.json', {'reason':task.get('stoppedReason'),
                'containers':[{k:c.get(k) for k in ('name','exitCode','reason')} for c in task['containers']]})
            s3.download_file(BUCKET,run_key+'/report.zip',str(output/'report.zip'))
            _extract(output/'report.zip',output/'report')
            completed = read_json(output/'report/completion.json') if (output/'report/completion.json').exists() else {}
            inference_exit = next(c.get('exitCode') for c in task['containers'] if c['name']=='inference')
            if inference_exit != 0 or not completed.get('complete'):
                raise RuntimeError('evaluation incomplete; previous baseline preserved; inspect report')
            # Workflow shares deployment concurrency, so no other staging evaluation races this pointer.
            s3.put_object(Bucket=BUCKET,Key=previous_key,Body=(output/'report/result.json').read_bytes())
            write_json(output/'published.json', {'report_key':run_key+'/report.zip','baseline_key':previous_key})
        finally:
            if task_arn:
                task = ecs.describe_tasks(cluster=STAGING_CLUSTER,tasks=[task_arn])['tasks'][0]
                if task['lastStatus'] != 'STOPPED':
                    ecs.stop_task(cluster=STAGING_CLUSTER,task=task_arn,reason='bounded staging evaluation cleanup')
            ecs.deregister_task_definition(taskDefinition=registered)


if __name__ == '__main__':
    main()
