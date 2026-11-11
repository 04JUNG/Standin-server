"""Select audited shoulder corrections and explicitly qualify upper-body anchors."""
import argparse,json,shutil
from pathlib import Path

def choose_mode(local,chest):
 """Prefer less motion unless chest repair makes a material shoulder improvement."""
 l=local['geometry_accepted'];c=chest['geometry_accepted'];le=local['diagnostics']['after']['rmse'];ce=chest['diagnostics']['after']['rmse']
 if c and (not l or ce<le-1):return 'chest'
 if l:return 'clavicle'
 return None

def anchor_policy(metrics):
 d=metrics['diagnostics'];m=d['after'] if metrics['geometry_accepted'] else d['before'];ready=max(m['endpoint_errors'])<=4 and m['vector_error']<=5 and m['center_error']<=3
 return dict(upper_body_fit_verified=ready,reason='shoulder_line_within_tolerance' if ready else 'shoulder_line_unresolved',allow_bounded_chest_in_future_joint_solve=not ready,lower_body_may_run=True,whole_upper_body_correctness_certified=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;rows=[]
 for c in [x['case'] for x in json.loads((r/'inputs.json').read_text())]:
  lm=json.loads((r/'clavicle-audited'/c/'fit/metrics.json').read_text());cm=json.loads((r/'chest-audited'/c/'fit/metrics.json').read_text());mode=choose_mode(lm,cm);chosen=mode or 'clavicle';src=r/(chosen+'-audited')/c;dest=r/'final'/c;shutil.copytree(src,dest);m=cm if chosen=='chest' else lm;m['shoulder_mode_selected']=mode;m['anchor_policy']=anchor_policy(m);(dest/'fit/metrics.json').write_text(json.dumps(m,indent=2));rows.append(dict(case_id=c,status='fit',geometry_accepted=m['geometry_accepted'],mode=mode,anchor_policy=m['anchor_policy']))
 (r/'final/results.json').write_text(json.dumps(rows,indent=2));print(rows)
if __name__=='__main__':main()
