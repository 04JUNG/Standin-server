"""Independent source, path, pose and FBX audit; no authoring source is written."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

SIZES = [-1., -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2.]
POSES = ['T', 'ARMS_DOWN', 'RAISE', 'BEND']


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def invariants(sb):
    m, a = sb.objects()
    s = sb.source()
    values = {'basis': s['co'].astype('<f4').tobytes(),
        'topology': s['faces'].astype('<i4').tobytes(),
        'weights': s['W'].astype('<f8').tobytes(),
        'uv': b''.join(np.array([v.uv[:] for v in layer.data], dtype='<f4').tobytes() for layer in m.data.uv_layers),
        'rest': canonical(s['rest']).encode(),
        'bones': canonical([(b.name, b.parent.name if b.parent else None, list(b.head_local), list(b.tail_local),
                            [list(row) for row in b.matrix_local]) for b in a.data.bones]).encode(),
        'materials': canonical([(slot.name, slot.diffuse_color[:], slot.use_nodes) for slot in m.data.materials]).encode()}
    return {k: sha_bytes(v) for k, v in values.items()}


def load(blend, controller, name):
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    spec = importlib.util.spec_from_file_location(name, controller)
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    sb.register()
    return sb


def pairs(q, faces):
    tree = BVHTree.FromPolygons(q.tolist(), faces.tolist(), all_triangles=True)
    return {(i, j) for i,j in tree.overlap(tree) if i < j and not set(faces[i]).intersection(faces[j])}


def metrics(sb, q, ref, s):
    f = s['faces']
    cross = np.cross(q[f[:, 1]]-q[f[:, 0]], q[f[:, 2]]-q[f[:, 0]])
    volume = float(np.einsum('ij,ij->', q[f[:, 0]], np.cross(q[f[:, 1]], q[f[:, 2]]))/6)
    x,y,z = s['co'].T
    roi = (abs(x)>.03)&(abs(x)<.14)&(z>1.20)&(z<1.39)&(y<-.03)
    ids = np.flatnonzero(roi & (x > 0))
    apex = ids[np.argmin(q[ids, 1])]
    return {**sb.geometry_metrics(q, ref, f), 'signed_volume_m3':volume,
            'min_triangle_area_mm2':float(np.linalg.norm(cross,axis=1).min()*5e5),
            'apex_vertex':int(apex),'apex_mm':(q[apex]*1000).tolist()}


def main(args):
    out = args.candidate.resolve()
    meta = json.loads((out/'candidate.json').read_text())
    for path,key in [(args.parent_blend,'blend_sha256'),(args.parent_controller,'controller_sha256')]:
        if sha_bytes(path.read_bytes()) != meta['parent'][key]:
            raise ValueError('Regression baseline does not match the candidate parent')
    report = {'rules':meta['rules'],'source':meta['source'],'poses':POSES,'sizes':SIZES,
              'invariants':{},'neutral':[],'combinations':[],'pose_checks':[],'exports':[],
              'limitations':['CSP has not been tested; artist signoff remains.',
              'BVH overlap is a broad-phase triangle test; inherited pairs are reported separately.']}
    data = {}
    samples = {}
    parent_shape = {}
    base_pair_sets = {}
    # Representative independent axes plus coupled thorax and breast extremes.
    combinations = [{}]
    for key, endpoints in {'physique':[-1,1.4], 'fat_gain':[2], 'muscle':[2,4],
                          'chest_width':[-1,1], 'chest_depth':[-1,1],
                          'breast_height':[-1,1], 'breast_width':[-1,1],
                          'breast_projection':[-1,1], 'breast_spacing':[-1,1]}.items():
        combinations += [{key:v} for v in endpoints]
    combinations += [{'chest_width':w,'chest_depth':d,'breast_spacing':sp,
                      'breast_width':bw,'breast_height':h,'breast_projection':p}
                     for w,d,sp,bw,h,p in [(-1,-1,-1,1,-1,-1),(-1,-1,-1,1,1,1),
                     (1,1,1,-1,-1,1),(1,-1,-1,1,1,-1),(-1,1,1,-1,1,1)]]
    combinations += [{'physique':1.4,'fat_gain':2,'breast_projection':1,'breast_spacing':-1},
                     {'physique':-1,'muscle':4,'chest_width':-1,'breast_width':1}]
    for label, blend, controller in [('old', args.parent_blend, args.parent_controller),
                                    ('new', Path(meta['source']['blend']), Path(meta['source']['controller']))]:
        sb = load(blend.resolve(), controller.resolve(), label)
        m,a = sb.objects()
        s = sb.source()
        default = sb.defaults(m)
        report['invariants'][label] = invariants(sb)
        data['faces'] = s['faces']
        if label == 'old':
            parent_deltas = {k:v.copy() for k,v in s['deltas'].items()}
            old_snapshot = sb.snapshot()
        else:
            report['unowned_keys_exact'] = all(np.array_equal(v,s['deltas'][k]) for k,v in parent_deltas.items() if k != 'BODY_BreastSmall')
            migrated = sb.migrate_breast_snapshot(old_snapshot)
            report['migration_controls_exact'] = migrated['controls']==old_snapshot['controls']
        for value in SIZES:
            c = {**default,'breast_size':min(1.,value) if label=='old' else value}
            q,_,ref = sb.compute(c)
            met = metrics(sb,q,ref,s)
            pp = pairs(q,s['faces'])
            if label=='old':
                parent_shape[value]=q.copy()
                base_pair_sets[value]=pp
            report['neutral'].append({'source':label,'size':value,**met,
                'pairs':len(pp),'new_pairs':len(pp-base_pair_sets[value]),
                'old_delta_max_mm':float(np.linalg.norm(q-parent_shape[value],axis=1).max()*1000)})
            samples[label,value]=q.copy()
        # One centi-unit grid sweep verifies the full interpolation path.
        sweep = []
        for v in np.arange(-100,201 if label=='new' else 101)/100:
            q,_,ref = sb.compute({**default,'breast_size':float(v)})
            met = metrics(sb,q,ref,s)
            if not met['finite'] or met['degenerate_faces'] or met['reversed_faces']:
                sweep.append({'size':float(v),**met})
        report[label+'_grid_failures'] = sweep
        for ci, overrides in enumerate(combinations):
            for size in [-1.,-.5,0.,1.,1.5,2.]:
                c = {**default,**overrides,'breast_size':min(1.,size) if label=='old' else size}
                q,_,ref = sb.compute(c)
                pp = pairs(q,s['faces'])
                key = ci,size
                if label=='old':base_pair_sets[key]=pp
                report['combinations'].append({'source':label,'case':ci,'size':size,'overrides':overrides,
                    **metrics(sb,q,ref,s),'pairs':len(pp),'new_pairs':len(pp-base_pair_sets[key])})
        for pose in POSES:
            for vi,size in enumerate(SIZES):
                c = {**default,'breast_size':min(1.,size) if label=='old' else size}
                sb.set_controls(c,correct=False)
                sb.apply_pose(pose)
                q = sb.evaluated(m)
                pp = pairs(q,s['faces'])
                key = pose,size
                if label=='old':base_pair_sets[key]=pp
                data[f'{label}_{pose}_{vi}'] = q.astype('<f4')
                qa = json.loads(m.get('surface_qa','{}'))
                report['pose_checks'].append({'source':label,'pose':pose,'size':size,'finite':bool(np.isfinite(q).all()),
                    'pairs':len(pp),'new_pairs':len(pp-base_pair_sets[key]),'surface_qa':qa,
                    'bone_pose_sha256':sha_bytes(canonical({b.name:[list(row) for row in b.matrix] for b in a.pose.bones}).encode()),
                    'intersection_calls_in_controller':sb.SURFACE_INTERSECTION_CALLS})
                if label=='new' and size in [-1.,0.,1.,1.5,2.] and pose in ['T','ARMS_DOWN']:
                    bundle=sb.export_fixed(str(out/'exports'),posed=pose!='T')
                    manifest=json.loads((bundle/'manifest.json').read_text())
                    report['exports'].append({'pose_name':pose,'size':size,'bundle':str(bundle),**manifest})
                print('BREAST_POSE_QA',label,pose,size,len(pp),flush=True)
        sb.unregister()
    report['invariants_equal'] = report['invariants']['old']==report['invariants']['new']
    report['positive_0_to_1_exact'] = all(np.array_equal(samples['old',v],samples['new',v]) for v in [0.,.25,.5,.75,1.])
    report['negative_path_monotonic_volume'] = all(a<=b+1e-12 for a,b in zip(
        [r['signed_volume_m3'] for r in report['neutral'] if r['source']=='new'],
        [r['signed_volume_m3'] for r in report['neutral'] if r['source']=='new'][1:]))
    report['success'] = (report['invariants_equal'] and report['unowned_keys_exact'] and report['positive_0_to_1_exact']
        and not report['new_grid_failures'] and all(r['new_pairs']==0 and r['finite'] and r['reversed_faces']==0
        and r['degenerate_faces']==0 for r in report['neutral']+report['combinations'] if r['source']=='new')
        and all(r['new_pairs']==0 and r['finite'] for r in report['pose_checks'] if r['source']=='new'))
    (out/'qa.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'review-data.npz',**data)
    print('BREAST_QA_COMPLETE',report['success'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--parent-blend',type=Path,required=True)
    ap.add_argument('--parent-controller',type=Path,required=True)
    main(ap.parse_args(sys.argv[sys.argv.index('--')+1:]))
