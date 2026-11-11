"""Exercise the frozen pose solver plus new authoring rig on a real library BVH."""
import argparse,json,sys,time,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import bpy
from converter import retarget as rt
from converter.shape_bake import convert_shaped,import_full_character

def main(args):
    release=Path(args.release);registry=json.loads((release/'shape-bakes.local.json').read_text(encoding='utf-8'))
    cid='male-base-v0-8-0' if args.gender=='male' else 'female-base-v0-4-0'
    row=registry['characters'][cid];recipe={**row,**registry['sources'][row['rules']],'values':row['defaults'],'shape_sha256':None}
    target=release/'conversion-qa'/(args.gender+'-'+Path(args.bvh).stem.replace(' ','_'));target.mkdir(parents=True,exist_ok=True)
    temporary=target/'tmp';temporary.mkdir(exist_ok=True);tempfile.tempdir=str(temporary)
    rt.import_character=import_full_character;t=time.time()
    report=convert_shaped(recipe=recipe,bvh_path=args.bvh,character_fbx=str(release/'fbx'/(cid+'.fbx')),out_path=str(target/'model.fbx'),
        frame=0,mirror=False,output_mode='rigged_rest',apply_root_translation=False,embed_textures=False,force_exact_v324=False)
    data=report.as_dict();data['elapsed_seconds']=time.time()-t
    (target/'result.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    if not report.ok:raise ValueError(data)
    print('CONVERSION_PASS',cid,round(data['elapsed_seconds'],2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--release',required=True);ap.add_argument('--gender',required=True);ap.add_argument('--bvh',required=True)
    main(ap.parse_args(sys.argv[sys.argv.index('--')+1:]))
