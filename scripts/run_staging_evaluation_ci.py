#!/usr/bin/env python3
"""Launch bounded evaluation shards using the current staging service definition."""
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
EVALUATOR_FILES = ('staging_live.py', 'staging_compare.py', 'rubric_judge.py', 'rubric_score.py',
                   'rubric_v32.py', 'rubric_v32_io.py', 'rubric_package.py', 'matching.py',
                   'schemas.py', 'http_runner.py', 'util.py', 'staging_merge.py')


def evaluator_fingerprint(root=ROOT):
    """Product search/refine modules shipped in the archive must not reset baseline."""
    paths = [root/'standin_eval'/name for name in EVALUATOR_FILES]
    paths += sorted((root/'evaluation/rubrics').rglob('*.json'))
    content = [(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


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


def failure_report(output, failures):
    """Keep incomplete live runs reviewable without publishing a score baseline."""
    rows = []
    for folder in sorted((output/'shards').glob('*/report')):
        for path in sorted((folder/'errors').glob('*.json')):
            rows.append((folder.parent.name, path.stem, 'artifact/refine', read_json(path).get('error_type')))
        for arm in ('search', 'refine'):
            path = folder/arm/'judge-traces.jsonl'
            if path.exists():
                for line in path.read_text().splitlines():
                    item = json.loads(line)
                    if item.get('status') == 'judge_error':
                        rows.append((folder.parent.name, item['case_id'], arm+' judge', item.get('error_type')))
    report = output/'report'; report.mkdir(exist_ok=True)
    write_json(report/'completion.json', {'complete':False, 'failed_shards':failures})
    text = ['# 스테이징 고정 50컷 평가 실패', '',
            '기준선은 갱신하지 않았습니다. 아래 오류와 shards/의 개별 보고서를 확인하세요.', '',
            '| 묶음 | 컷 | 단계 | 오류 |', '|---|---|---|---|']
    text += ['| '+' | '.join(map(str, row))+' |' for row in rows]
    if not rows:
        text.append('기동 또는 보고서 수집 실패: failures.json과 task-stop.json을 확인하세요.')
    (report/'report.md').write_text('\n'.join(text)+'\n')


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
        contract = hashlib.sha256((evaluator_fingerprint()+args.judge_model+DATA_SHA).encode()).hexdigest()
        series = PREFIX+'/'+contract
        run_key = series+'/runs/'+args.run_id
        s3.upload_file(str(archive), BUCKET, run_key+'/code.zip')
        def signed(method, key):
            return s3.generate_presigned_url(method, Params={'Bucket':BUCKET,'Key':key}, ExpiresIn=3300)
        previous_key = series+'/baseline.json'
        previous_url = ''
        previous_value = None
        previous_etag = None
        existing = s3.list_objects_v2(Bucket=BUCKET, Prefix=previous_key, MaxKeys=1).get('Contents', [])
        if any(item['Key'] == previous_key for item in existing):
            previous_object = s3.get_object(Bucket=BUCKET, Key=previous_key)
            previous = previous_object['Body'].read()
            previous_etag = previous_object['ETag']
            # Snapshot previous baseline: never depend on a mutable URL during execution.
            s3.put_object(Bucket=BUCKET, Key=run_key+'/previous.json', Body=previous)
            previous_url = signed('get_object', run_key+'/previous.json')
            previous_value = json.loads(previous)
        values = {'EVAL_CODE_URL':signed('get_object',run_key+'/code.zip'), 'EVAL_CODE_SHA256':code_hash,
            'EVAL_DATA_URL':signed('get_object',DATA_KEY), 'EVAL_DATA_SHA256':DATA_SHA,
            'EVAL_RESULT_URL':signed('put_object',run_key+'/report.zip'), 'EVAL_PREVIOUS_URL':previous_url,
            'EVAL_REVISION':args.revision,'EVAL_JUDGE_MODEL':args.judge_model,'EVAL_CONTRACT_VERSION':contract}
        registered = ecs.register_task_definition(**evaluation_definition(definition))['taskDefinition']['taskDefinitionArn']
        task_arns = []
        try:
            for shard in range(3):
                shard_values = {**values, 'EVAL_SHARD_INDEX':str(shard), 'EVAL_SHARD_COUNT':'3',
                    'EVAL_RESULT_URL':signed('put_object',run_key+f'/shard-{shard}.zip')}
                response = ecs.run_task(cluster=STAGING_CLUSTER, taskDefinition=registered, launchType='FARGATE',
                    networkConfiguration=service['networkConfiguration'], count=1, startedBy='fixed50-staging-eval',
                    tags=[{'key':'StandinEvaluation','value':'fixed50'}],
                    overrides={'containerOverrides':[{'name':'inference','environment':[{'name':k,'value':v} for k,v in shard_values.items()]}]})
                task_arns.extend(t['taskArn'] for t in response.get('tasks', []))
                if response.get('failures') or not response.get('tasks'):
                    raise RuntimeError('evaluation task launch failed')
            write_json(output/'tasks.json', {'tasks':task_arns,'revision':args.revision,'dataset':DATA_SHA,'contract':contract})
            deadline = time.monotonic()+args.timeout_seconds
            while time.monotonic() < deadline:
                tasks = ecs.describe_tasks(cluster=STAGING_CLUSTER,tasks=task_arns)['tasks']
                if len(tasks)==3 and all(t['lastStatus']=='STOPPED' for t in tasks):
                    break
                print(json.dumps({'statuses':[t['lastStatus'] for t in tasks],
                    'elapsed_seconds':int(args.timeout_seconds-(deadline-time.monotonic()))}),flush=True)
                time.sleep(20)
            else:
                raise TimeoutError('staging evaluation deadline exceeded')
            stopped = {t['taskArn']:t for t in tasks}
            fragments = []; failures = []
            for shard, arn in enumerate(task_arns):
                task = stopped[arn]
                folder = output/'shards'/str(shard); folder.mkdir(parents=True)
                write_json(folder/'task-stop.json', {'reason':task.get('stoppedReason'),
                    'containers':[{k:c.get(k) for k in ('name','exitCode','reason')} for c in task['containers']]})
                try:
                    s3.download_file(BUCKET,run_key+f'/shard-{shard}.zip',str(folder/'report.zip'))
                    _extract(folder/'report.zip',folder/'report')
                    completed = read_json(folder/'report/completion.json')
                    inference_exit = next(c.get('exitCode') for c in task['containers'] if c['name']=='inference')
                    if inference_exit != 0 or not completed.get('complete'):
                        raise RuntimeError('incomplete shard')
                    fragments.append(folder/'report')
                except Exception as exc:
                    failures.append({'shard':shard,'error_type':type(exc).__name__})
            if failures:
                write_json(output/'failures.json',failures)
                failure_report(output, failures)
                raise RuntimeError('evaluation incomplete; previous baseline preserved; inspect shards')
            from standin_eval.staging_merge import merge_fragments
            import shutil
            merge_fragments(fragments,output/'report',previous_value)
            archive = shutil.make_archive(str(output/'report'),'zip',output/'report')
            s3.upload_file(archive,BUCKET,run_key+'/report.zip')
            # Also protect against a separately launched manual run finishing out of order.
            condition = {'IfMatch':previous_etag} if previous_etag else {'IfNoneMatch':'*'}
            s3.put_object(Bucket=BUCKET,Key=previous_key,Body=(output/'report/result.json').read_bytes(),**condition)
            write_json(output/'published.json', {'report_key':run_key+'/report.zip','baseline_key':previous_key})
        finally:
            if task_arns:
                for task in ecs.describe_tasks(cluster=STAGING_CLUSTER,tasks=task_arns)['tasks']:
                    if task['lastStatus'] != 'STOPPED':
                        ecs.stop_task(cluster=STAGING_CLUSTER,task=task['taskArn'],reason='bounded staging evaluation cleanup')
            ecs.deregister_task_definition(taskDefinition=registered)


if __name__ == '__main__':
    main()
