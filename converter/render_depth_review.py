"""Render exact before/after FBXs and extract actual rig joints for local review."""
from pathlib import Path
import argparse,json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    from converter.thumbnail_render import render_artifact_view
    from converter.refine_v3_safe_snapshot_qa import snapshot
    from converter.shape_bake import import_full_character
    from converter import retarget as rt
    from converter.pose_safety import COCO_NAMES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,path in [('before',a.before),('after',a.after)]:
        renders=[]
        for view in ('front','three_quarter','side'):
            renders.append(render_artifact_view(artifact_fbx=path,output_png=a.output_dir/(label+'-'+view+'.png'),view=view,resolution=640,samples=12,engines=('CYCLES',)))
        rt.import_character=import_full_character;s=snapshot(path)
        ids=[s['names'].index('mixamorig:'+n) for n in COCO_NAMES]
        rows.append(dict(label=label,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),joints_world=s['bones'][ids,:3,3].tolist(),renders=renders))
    (a.output_dir/'actual-joints.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
