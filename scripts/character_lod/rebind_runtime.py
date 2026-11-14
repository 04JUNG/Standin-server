"""Rebind only the byte container after proven same-topology dependency cleanup.

This is a newly sampled CANDIDATE SSR2, never the original release's SSR2. All
arrays and controller bytes remain identical. Independent validation must still
run on the exact final source and may (and does) reject preview promotion.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/character_upgrade'))
from check_sampled_runtime import read_ssr2
from build_sampled_runtime import write_ssr2
p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);a=p.parse_args();folder=Path(a.candidate);m=json.loads((folder/'manifest.json').read_text());h,arrays=read_ssr2(a.runtime);old=dict(h['source'])
for name,digest in m['files'].items():
 if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('candidate file changed')
if old['blendSha256']!=m['container_cleanup']['source_blend_sha256'] or old['controllerSha256']!=m['files']['controller.py']:raise ValueError('different geometry/controller lineage')
h['source']={'blendSha256':m['files']['model.blend'],'controllerSha256':m['files']['controller.py']};specs=h.pop('arrays');write_ssr2(Path(a.out),h,[(s['name'],s['dtype'],arrays[s['name']]) for s in specs]);report={'original_candidate_runtime_sha256':hashlib.sha256(Path(a.runtime).read_bytes()).hexdigest(),'source_before':old,'source_after':h['source'],'semantic_sha256':m['container_cleanup']['semantic_sha256'],'runtime_sha256':hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),'bytes':Path(a.out).stat().st_size,'same_candidate_topology':True,'original_release_runtime_reused':False};Path(a.out+'.lineage.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
