"""Prepare, but do not deploy, a staging-only RC4 task definition overlay."""
import argparse, copy, json
from pathlib import Path

def prepare(raw, image, worker_image=None):
    task=copy.deepcopy(raw.get('taskDefinition',raw))
    if not task.get('family','').startswith('StandinStagingApp'):
        raise ValueError('RC4 activation is staging-only')
    if '@sha256:' not in image or (worker_image and '@sha256:' not in worker_image):
        raise ValueError('immutable image digests required')
    containers={c['name']:c for c in task['containerDefinitions']}
    def env(c,**values):
        current={e['name']:e['value'] for e in c.get('environment',[])};current.update(values)
        c['environment']=[dict(name=k,value=v) for k,v in sorted(current.items())]
    if 'inference' in containers:
        if not worker_image:raise ValueError('worker image required')
        main=containers['inference'];main['image']=image
        env(main,REFINE_RELEASE_ENABLED='1',REFINE_RELEASE_WORKER_URL='http://127.0.0.1:8002')
        secrets=[s for s in main.get('secrets',[]) if s['name']=='OPENAI_API_KEY']
        if len(secrets)!=1:raise ValueError('existing staging OpenAI secret required')
        logging=copy.deepcopy(main['logConfiguration'])
        if logging['logDriver']!='awslogs':raise ValueError('worker logging configuration required')
        logging['options']['awslogs-stream-prefix']='rc4-worker'
        worker=dict(name='refine-rc4',image=worker_image,essential=True,cpu=3072,memoryReservation=2048,environment=[dict(name='REFINE_RELEASE_ENABLED',value='1')],secrets=secrets,logConfiguration=logging,
            healthCheck=dict(command=['CMD-SHELL',"python -c \"import urllib.request;urllib.request.urlopen('http://127.0.0.1:8002/refine/release/ready',timeout=4)\""],interval=30,timeout=5,retries=3,startPeriod=60))
        task['containerDefinitions']=[c for c in task['containerDefinitions'] if c['name']!='refine-rc4']+[worker]
        task['cpu']='4096';task['memory']=str(max(8192,int(task['memory'])))
    elif 'bff' in containers:
        containers['bff']['image']=image;env(containers['bff'],REFINE_RELEASE_ENABLED='1')
    else:raise ValueError('unexpected staging task')
    for k in ['taskDefinitionArn','revision','status','requiresAttributes','compatibilities','registeredAt','registeredBy','deregisteredAt']:task.pop(k,None)
    return task

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',type=Path);p.add_argument('--image',required=True);p.add_argument('--worker-image');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with a.out.open('x') as f:json.dump(prepare(json.loads(a.task.read_text()),a.image,a.worker_image),f,indent=2)
