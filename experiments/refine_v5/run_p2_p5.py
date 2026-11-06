"""Reproduce the complete P2-P5 offline experiment from saved P0 data.
No service, paid provider, deployment, or P6 test execution is included.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--out',required=True);p.add_argument('--body',required=True);p.add_argument('--blender',required=True);p.add_argument('--epochs',type=int,default=140);p.add_argument('--variants',type=int,default=32);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];previous=Path(a.previous).resolve();out=Path(a.out).resolve();out.mkdir(exist_ok=False,parents=True)
    for name in ['data/native_audit.json','all-seeds/seeds.json','data/러프','data/정답_3d인형']:
        if not (previous/name).exists():raise FileNotFoundError(previous/name)
    stages=[]
    def stage(name,command,expected):
        started=time.monotonic()
        with (out/(name+'.log')).open('w') as log:result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=False)
        item=dict(stage=name,returncode=result.returncode,seconds=time.monotonic()-started,expected=str(expected));stages.append(item)
        (out/'stages.json').write_text(json.dumps(stages,indent=2))
        if result.returncode or not Path(expected).exists():raise RuntimeError('stage failed: '+name)
        print('V5_STAGE',name,'complete',flush=True)
    config=dict(repo=str(repo),out=str(out/'native-data'),seeds=str(previous/'all-seeds/seeds.json'),split=str(repo/'evaluation/refine_v5/split-v2.json'),data=str(previous/'data'),body=str(Path(a.body).resolve()))
    cfg=out/'prepare.json';cfg.write_text(json.dumps(config,indent=2))
    def blender(script,*args):return [a.blender,'-b','--python-exit-code','1','--python',str(repo/'experiments/refine_v5'/script),'--',*map(str,args)]
    def python(module,*args):return [sys.executable,'-m','experiments.refine_v5.'+module,*map(str,args)]
    stage('prepare',blender('prepare_learning_native.py',cfg),out/'native-data/records.json')
    native=json.loads((out/'native-data/records.json').read_text())
    if any(r['status']!='ready' for r in native):raise RuntimeError('native preparation incomplete; inspect records')
    common=['--run',out,'--previous',previous,'--repo',repo]
    stage('dataset',python('learning_data',*common,'--variants',a.variants),out/'dataset/manifest.json')
    stage('train',python('train_v3','--run',out,'--epochs',a.epochs),out/'models/selection.json')
    for fusion,without,directory in [('observation',True,'iterative-geometric'),('observation',False,'iterative'),('learned_balanced',False,'iterative-balanced')]:
        args=common+['--body',a.body,'--fusion',fusion]+(['--without-learning'] if without else [])
        stage(directory,python('run_iterative',*args),out/directory/'manifest.json')
        evalfusion='geometric' if without else fusion
        filename={'geometric':'evaluation-geometric.json','observation':'evaluation.json','learned_balanced':'evaluation-balanced.json'}[evalfusion]
        stage('evaluate-'+evalfusion,python('evaluate_v3',*common,'--fusion',evalfusion),out/filename)
    stage('frame-evaluation',python('evaluate_frames','--run',out),out/'frame-evaluation.json')
    stage('native-audit',blender('audit_iterative_native.py',out),out/'native-audit/records.json')
    stage('readback',blender('readback_v3.py',out),out/'native-audit/independent-readback.json')
    stage('review',python('render_review','--run',out,'--previous',previous),out/'native-review.png')
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (out/'reproduction.json').write_text(json.dumps(dict(offline_only=True,test_used=False,body_sha256=sha(a.body),code_sha256={str(x.relative_to(repo)):sha(x) for x in (repo/'experiments/refine_v5').glob('*.py')},stages=stages),indent=2))
if __name__=='__main__':main()
