"""Coupled body/pose regression against the pinned legacy source, in Blender."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify import load, pairs


def main(args):
    out = args.candidate.resolve()
    meta = json.loads((out/'candidate.json').read_text())
    for path, key in [(args.parent_blend, 'blend_sha256'), (args.parent_controller, 'controller_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['parent'][key]:
            raise ValueError('Regression baseline does not match the candidate parent')
    combos = [
        {'physique':1.4,'fat_gain':2,'breast_spacing':-1,'breast_projection':1},
        {'physique':-1,'muscle':4,'chest_width':-1,'breast_width':1},
        {'chest_width':-1,'chest_depth':-1,'breast_height':-1,
         'breast_width':1,'breast_spacing':-1,'breast_projection':-1},
    ]
    reference, rows = {}, []
    for label, blend, controller in [
        ('old',args.parent_blend,args.parent_controller),
        ('new',Path(meta['source']['blend']),Path(meta['source']['controller'])),
    ]:
        sb = load(blend.resolve(), controller.resolve(), label)
        mesh,_ = sb.objects()
        s = sb.source()
        default = sb.defaults(mesh)
        cent = s['co'][s['faces']].mean(1)
        # The version-pinned 1.75 m rest frame: breast, lower pole and front axilla.
        breast = (abs(cent[:,0])<.18)&(cent[:,1]<.02)&(cent[:,2]>1.18)&(cent[:,2]<1.41)
        for ci, controls in enumerate(combos):
            for size in [-1.,2.]:
                for pose in ['ARMS_DOWN','RAISE','BEND']:
                    sb.set_controls({**default,**controls,
                        'breast_size':min(size,1.) if label=='old' else size},correct=False)
                    sb.apply_pose(pose)
                    q = sb.evaluated(mesh)
                    pp = pairs(q,s['faces'])
                    key = ci,size,pose
                    if label=='old':
                        reference[key] = pp
                    added = pp-reference[key]
                    local = [(i,j) for i,j in added if breast[i] or breast[j]]
                    row = {'source':label,'case':ci,'controls':controls,'size':size,'pose':pose,
                        'pairs':len(pp),'new_pairs':len(added),'new_breast_pairs':len(local),
                        'finite':bool(np.isfinite(q).all())}
                    rows.append(row)
                    print(json.dumps(row),flush=True)
        sb.unregister()
    result = {'rows':rows,'no_added_breast_pairs':all(r['new_breast_pairs']==0 for r in rows),
              'no_added_global_pairs':all(r['new_pairs']==0 for r in rows)}
    (out/'pose-stress.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--parent-blend',type=Path,required=True)
    ap.add_argument('--parent-controller',type=Path,required=True)
    main(ap.parse_args(sys.argv[sys.argv.index('--')+1:]))
