"""Hash final local deliverables without packaging working samples or deploying."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);a=p.parse_args();root=Path(a.assets);files=[]
for name in ('originals','compact','delivery-package','prepared-baseline-v2','prepared-candidates-compact','review','benchmark'):
 for path in sorted((root/name).rglob('*')):
  if not path.is_file() or '__pycache__' in path.parts:continue
  files.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
for sex in ('male','female'):
 for name in ('delivery.ssr','delivery.ssr.lineage.json','independent.json'):
  path=root/('runtime-'+sex)/name;files.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
report={'schema':'local-character-lod-review.v1','baseline_sha':'ee965494435dbaf26b643c2e1f8da1fe404dc513','scope':'local review only; no deployment or catalog replacement','promotable':False,'reasons':['extra conservative deformed-protected-point screen has held cases','both new SSR2 independent approximation checks fail','representative CSP imports do not certify every shape/skin'],'files':files};(root/'delivery-manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('FINAL_LOCAL_DELIVERABLES',len(files),'files',sum(x['bytes'] for x in files),'bytes',flush=True)
