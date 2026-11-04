"""Opt-in native part runner. No production API or search defaults are changed.

Blender --background --python-exit-code 1 --python scripts/refine_v4_parts.py --
    requests.json NEW_OUTPUT
Paths in requests are relative to the request file. Current asset fingerprints
and side attribution are mandatory; rejected edits return that exact asset.
"""
import shutil,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.refine_v4.runner import execute,sha,save


if __name__=='__main__':
    request_file,out=map(lambda v:Path(v).resolve(),sys.argv[sys.argv.index('--')+1:])
    requests=json.loads(request_file.read_text());ids=[r['id'] for r in requests]
    if len(set(ids))!=len(ids) or any(not s.replace('-','').isalnum() for s in ids):raise ValueError('unique simple request IDs required')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(request_file,out/'requests.json');rows=[]
    for req in requests:
        row=execute(req,request_file.parent,out/req['id']);row['after_sha256']=sha(out/req['id']/'after.fbx')
        save(out/req['id']/'receipt.json',row);rows.append(row);save(out/'summary.json',rows)
        print(req['id'],row['status'],row['reasons'],flush=True)
