"""Build a hash-pinned registry from prepared, unedited library assets."""
import argparse, json
from pathlib import Path
from .service import sha, inside

def register(assets, cases):
    source = assets.resolve(); manifest = source/'input/manifest.json'; result = {}
    for row in json.loads(manifest.read_text())['rows']:
        case = row['case_id']
        if case not in cases: continue
        if row.get('status')!='original_library_bvh_confirmed' or row.get('target_identity')!='single_depicted_target':
            raise ValueError('Unconfirmed original asset')
        rigpath=f'input/p1-native/{case}-source.json';rig=json.loads(inside(source,rigpath).read_text())
        files=[rigpath,f'base-mesh/{case}.npz','input/'+rig['source'],'input/'+row['selection']['verified_selected_bvh']]
        hashes={p:sha(inside(source,p)) for p in files}
        if hashes[files[-1]]!=rig['source_bvh_sha256'] or rig['source_bvh_sha256']!=row['selection']['selected_bvh_sha256'] or hashes[files[-2]]!=rig['source_sha256']:
            raise ValueError('Original BVH/FBX binding mismatch')
        pose=row['selection']['selected_pose_id']
        if pose in result: raise ValueError('Duplicate pose ID; register a single canonical asset per pose')
        result[pose]=dict(assets=str(source),case=case,manifest_sha256=sha(manifest),files=hashes)
    if len(result)!=len(cases): raise ValueError('Some requested assets are absent or duplicate')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=register(a.assets,a.cases)
    with a.out.open('x') as f: json.dump(data,f,indent=2)
