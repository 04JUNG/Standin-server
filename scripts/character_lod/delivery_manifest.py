"""Hash final local deliverables without packaging working samples or deploying."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--revision',choices=('initial','followup','hands-v4'),default='initial');a=p.parse_args();root=Path(a.assets);files=[]
names=('originals','compact','delivery-package','prepared-baseline-v2','prepared-candidates-compact','review','benchmark') if a.revision=='initial' else ('originals','compact','delivery-package','prepared-baseline','prepared-candidates','review','benchmark')
for name in names:
 for path in sorted((root/name).rglob('*')):
  if not path.is_file() or '__pycache__' in path.parts:continue
  files.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
for sex in ('male','female') if a.revision=='initial' else ():
 for name in ('delivery.ssr','delivery.ssr.lineage.json','independent.json'):
  path=root/('runtime-'+sex)/name;files.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
reasons=['extra conservative deformed-protected-point screen has held cases','both new SSR2 independent approximation checks fail','representative CSP imports do not certify every shape/skin'] if a.revision=='initial' else ['actual Converter closed-fist final correspondence screen remains held','human review and skin certification still required','new topology has no activated approximate SSR2 certification','earlier CSP checks belong to different candidates']
if a.revision=='hands-v4':
 pairs=json.loads((root/'benchmark/final-pairs.json').read_text())
 if len(pairs)!=12 or not all(row['numeric_screen_pass'] for row in pairs):raise ValueError('v4 requires all twelve final geometry pairs')
 reasons=['human review and skin certification still required','new topology has no activated approximate SSR2 certification','new candidate CSP import is unverified; picker Open disabled even for previous files']
report={'schema':'local-character-lod-review.v1','revision':a.revision,'baseline_sha':'ee965494435dbaf26b643c2e1f8da1fe404dc513','scope':'local review only; no deployment or catalog replacement','promotable':False,'reasons':reasons,'files':files};(root/'delivery-manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('FINAL_LOCAL_DELIVERABLES',len(files),'files',sum(x['bytes'] for x in files),'bytes',flush=True)
