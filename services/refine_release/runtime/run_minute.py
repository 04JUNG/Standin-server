"""One original image and manifest-bound original pose -> paired candidates -> verified release."""
import argparse,json,os,sys,time,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;EXP=ROOT/'experiments'
def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--extractor-checkout',type=Path,required=True);p.add_argument('--model-manifest',type=Path,required=True);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);t0=time.monotonic();events=[];c=a.case
 env=dict(os.environ,POSE_CANARY_STAGE='shadow',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',REFINE_MINUTE_STATE=str(a.out/'shared-compute'))
 def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2))
 def run(label,args):
  t=time.monotonic()
  with (a.out/(label+'.log')).open('w') as log:r=subprocess.run(list(map(str,args)),env=env,stdout=log,stderr=subprocess.STDOUT)
  events.append(dict(stage=label,seconds=time.monotonic()-t,returncode=r.returncode));write(a.out/'timing-progress.json',events)
  if r.returncode and not (label=='skeleton' and r.returncode==42):raise RuntimeError(label+' failed')
  return r.returncode
 prepared=a.out/'prepared';skeleton_code=run('skeleton',[sys.executable,EXP/'refine_new/extract_develop.py','--checkout',a.extractor_checkout,'--model-manifest',a.model_manifest,'--assets',a.assets,'--out',prepared/c/'observations','--cases',c]);write(prepared/c/'skeleton-status.json',dict(case=c,status='complete',seconds=events[-1]['seconds']))
 if skeleton_code==42:
  decision=dict(case=c,status='review_required',release_eligible=False,selected=None,reason='insufficient_skeleton_evidence',new_vlm_calls=0,diagnostic=json.loads((prepared/c/'observations'/f'{c}-diagnostic.json').read_text()))
  write(a.out/'gate'/c/'decision.json',decision);write(a.out/'gate/summary.json',[decision]);write(a.out/'timing.json',dict(events=events,wall_seconds=time.monotonic()-t0,new_vlm_calls=0));write(prepared/c/'skeleton-status.json',dict(case=c,status='withheld',reason=decision['reason']));return
 # The extraction process uses the same BLAS limits as every local subprocess.
 run('vlm_once',[sys.executable,ROOT/'extract_once.py','--assets',a.assets,'--prepared',prepared,'--case',c])
 (a.out/'relations').mkdir();shutil.copy2(prepared/c/'semantics/relations.json',a.out/'relations'/f'{c}.json')
 run('paired_generation',[sys.executable,EXP/'refine_unified/compare_cohort.py','--assets',a.assets,'--prepared',prepared,'--fresh-prepared',prepared,'--out',a.out/'paired','--extractor-checkout',a.extractor_checkout,'--model-manifest',a.model_manifest,'--cases',c,'--workers','1'])
 run('final_gate',[sys.executable,ROOT/'finalize.py','--run',a.out/'paired','--assets',a.assets,'--relations-dir',a.out/'relations','--out',a.out/'gate','--cases',c])
 elapsed=time.monotonic()-t0;write(a.out/'timing.json',dict(version='New Refine Release RC3',case=c,wall_seconds=elapsed,events=events,new_vlm_calls=1,new_skeleton=True,source='manifest_bound_unedited_original',manual_conditions_used=False,search_rerun=False,within_75_seconds=elapsed<=75,time_target_seconds=60));print(c,round(elapsed,2),json.loads((a.out/'gate/summary.json').read_text())[0].get('selected'),flush=True)
if __name__=='__main__':main()
