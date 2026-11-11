"""Audit cases independently so one export failure cannot hide other outcomes."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--blender',required=True);a=p.parse_args();root=a.run.resolve();rows=json.loads((root/'results.json').read_text());audits=[]
for row in rows:
 c=row['case_id'];folder=root/'audit-cases'/c;folder.mkdir(parents=True,exist_ok=False);(folder/c).symlink_to(root/c,target_is_directory=True);(folder/'results.json').write_text(json.dumps([row]))
 with (folder/'log.txt').open('w') as log:r=subprocess.run([a.blender,'-b','--python-exit-code','1','--python',str(Path(__file__).resolve().parent/'audit_blender.py'),'--',str(folder),str(a.baseline.resolve()),str(a.assets.resolve())],stdout=log,stderr=subprocess.STDOUT)
 meta=json.loads((root/c/'fit/metrics.json').read_text());row['geometry_accepted']=meta['geometry_accepted']
 if (folder/'audit.json').exists():audit=json.loads((folder/'audit.json').read_text())[0]
 else:audit=dict(case_id=c,rollback=meta.get('collision_fallback',False),tested_crossing_increases=meta.get('collision_screen',{}).get('increases'),export=json.loads((root/c/'fit/export-check.json').read_text()) if (root/c/'fit/export-check.json').exists() else None)
 audit.update(export_passed=r.returncode==0,error=None if r.returncode==0 else 'See audit-cases log; export not validated');audits.append(audit);print(c,'accepted',row['geometry_accepted'],'export',r.returncode,flush=True)
(root/'results.json').write_text(json.dumps(rows,indent=2));(root/'audit.json').write_text(json.dumps(audits,indent=2));(root/'status.json').write_text(json.dumps(dict(status='audited_experimental' if all(x['export_passed'] for x in audits) else 'audited_with_export_failures',certified=False),indent=2))
