import argparse,sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'experiments/refine_unified'))
from semantics import extract_once
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--case',required=True);a=p.parse_args();r=next(r for r in json.loads((a.assets/'input/manifest.json').read_text())['rows'] if r['case_id']==a.case);t=time.monotonic();d=extract_once(a.assets/'input'/r['rough'],a.prepared/a.case/'observations'/f'{a.case}.json',a.prepared/a.case/'semantics');(a.prepared/'luna-summary.json').write_text(json.dumps([dict(case=a.case,status='complete',seconds=time.monotonic()-t)]));print('one combined call',d['seconds'])
