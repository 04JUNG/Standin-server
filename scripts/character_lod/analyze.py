"""Validate full repeated evidence and summarize real nested timing spans."""
import argparse,hashlib,json,statistics,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.character_lod.contract import require_paired_runs
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);a=p.parse_args();root=Path(a.assets);rows=json.loads((root/'benchmark/results.json').read_text());require_paired_runs(rows,3);assert len(rows)==72
stages=defaultdict(lambda:defaultdict(list));inventory=[]
for row in rows:
 path=Path(row['file'].replace('/work',str(root),1));assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];key=' / '.join(row[k] for k in ('gender','case','skin','variant'));values={k:row[k] for k in ('converter_seconds','skin_seconds','file_save_seconds','file_ready_seconds')};values['neutral_restore']=row['report']['shape_bake']['neutral_preparation']['seconds']
 for c in row['report']['local_lod_timing']['calls']:values[c['stage']]=values.get(c['stage'],0)+c['seconds']
 values['worker_total']=row['report']['local_lod_timing']['worker_total_seconds']
 for stage,value in values.items():stages[key][stage].append(value)
 inventory.append({k:row[k] for k in ('gender','case','skin','variant','trial','sha256','bytes','controls','bvh_sha256')})
summary={key:{stage:{'mean':statistics.mean(x),'median':statistics.median(x),'min':min(x),'max':max(x),'n':len(x)} for stage,x in values.items()} for key,values in stages.items()}
(root/'benchmark/stages.json').write_text(json.dumps({'nested_spans':True,'do_not_sum_stages':True,'groups':summary},indent=2)+'\n');(root/'benchmark/inventory.json').write_text(json.dumps(inventory,indent=2)+'\n');print('AUDITED_72_FINAL_FILES',flush=True)
