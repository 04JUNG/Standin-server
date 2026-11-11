"""One-command autonomous depth extension for new refine; no manual conditions argument."""
import argparse,sys,subprocess,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--blender',type=Path,required=True);p.add_argument('--observations-cache',type=Path);p.add_argument('--semantics-cache',type=Path);p.add_argument('--extractor-checkout',type=Path);p.add_argument('--model-manifest',type=Path);p.add_argument('--legacy-depth-only',action='store_true');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent;auto=here.parent/'refine006_auto'
obs=a.observations_cache or a.out/'observations';sem=a.semantics_cache or a.out/'semantics'
if not a.observations_cache:
 if not a.extractor_checkout or not a.model_manifest:raise ValueError('Fresh extraction needs checkout and model manifest')
 subprocess.run([sys.executable,str(here/'extract_develop.py'),'--checkout',str(a.extractor_checkout),'--model-manifest',str(a.model_manifest),'--assets',str(a.assets),'--out',str(obs),'--cases',*a.cases],check=True,env=dict(os.environ,POSE_CANARY_STAGE='shadow'))
if not a.semantics_cache:subprocess.run([sys.executable,str(auto/'extract.py'),'--assets',str(a.assets),'--out',str(sem),'--cases',*a.cases],check=True)
prior=a.out/('final' if a.legacy_depth_only else 'initial-depth')
subprocess.run([sys.executable,str(auto/'pipeline.py'),'--assets',str(a.assets),'--baseline',str(a.baseline),'--observations',str(obs),'--semantics',str(sem),'--out',str(prior),'--cases',*a.cases,'--blender',str(a.blender),'--skip-audit'],check=True)
def audit(folder):
 subprocess.run([sys.executable,str(auto/'audit_regions.py'),'--run',str(folder),'--baseline',str(a.baseline),'--assets',str(a.assets),'--blender',str(a.blender)],check=True)
 if not all(r['export_passed'] for r in json.loads((folder/'audit.json').read_text())):raise ValueError('Depth export failed')
audit(prior)
if not a.legacy_depth_only:
 subprocess.run([sys.executable,str(auto/'regional.py'),'--assets',str(a.assets),'--baseline',str(a.baseline),'--observations',str(obs),'--semantics',str(sem),'--accepted-prior',str(prior),'--out',str(a.out/'final'),'--cases',*a.cases],check=True)
 audit(a.out/'final')
(a.out/'run.json').write_text(json.dumps(dict(status='complete',observations=str(obs),semantics=str(sem),manual_conditions_used=False,reference_used_for_fit=False,production_enabled=False),indent=2))
print(a.out/'final')
