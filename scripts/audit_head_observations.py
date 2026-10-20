"""Independent camera-to-pixel reconstruction of head diagnostic locations."""
import argparse
import hashlib
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import sha256,save_json
from scripts.experiment_near_camera import read
from scripts.diagnose_head_observations import summary
from scripts.diagnose_near_preview import surface,capture
from standin_eval.perspective_search import Camera,project,camera_points


def run(args):
    policy=read(args.out/'protocol.json');result=read(args.out/'results.json')
    assert all(sha256(Path(p))==h for p,h in policy['hashes'].items())
    assert sha256(ROOT/'scripts/diagnose_head_observations.py')==policy['script_sha256']
    source=Path(policy['source']);answers={r['image']:r for r in read(source/'answers.json')}
    poses={p.pose_id:p for p in load_library(args.library_root)}
    manifest={r['image']:r for r in read(source/'manifest.json')}
    assert len(result['records'])==len(answers)==len(manifest)==360
    maxerr=0.;replayed=0
    for row in result['records']:
        a=answers[row['image']];gt=np.asarray(a['ground_truth_xy']);cam=Camera(**a['camera'])
        assert sha256(source/'images'/row['image'])==manifest[row['image']]['sha256']
        pose=poses[a['pose_id']];xy,v=project(pose.joints,cam);assert v
        # Recover ONLY the pixel scale/translation from source joint projections.
        xc=xy-xy.mean(axis=0);yc=gt-gt.mean(axis=0)
        scale=float((xc*yc).sum()/(xc*xc).sum());offset=gt.mean(axis=0)-scale*xy.mean(axis=0)
        assert np.max(abs(xy*scale+offset-gt))<.001
        p=camera_points(pose.joints,cam).astype(float)
        up=(p[5]+p[6])/2;up/=np.linalg.norm(up)
        lateral=p[5]-p[6];lateral/=np.linalg.norm(lateral)
        forward=np.cross(up,lateral);forward/=np.linalg.norm(forward)
        lateral=np.cross(forward,up);basis=np.column_stack((lateral,up,forward))
        def pixels(q):
            q=np.asarray(q);out=q[:,:2].copy();out[:,1]*=-1
            if cam.distance is not None:out/=q[:,2,None]+cam.distance
            return out*scale+offset
        expected=pixels((p[0]+.215*forward)[None])[0]
        error=float(np.max(abs(expected-row['nose_marker_pixels'])));maxerr=max(maxerr,error)
        assert error<.001
        pred=read(source/'predictions'/(Path(row['image']).stem+'.json'))
        if 'nose_score' not in row:continue
        obs=np.asarray(pred['keypoints2d'])[0];torso=np.linalg.norm((gt[5]+gt[6]-gt[11]-gt[12])/2)
        assert row['nose_score']==float(pred['scores'][0])
        assert abs(np.linalg.norm(obs-gt[0])/torso-row['error_to_head_center_gt_torso'])<1e-5
        assert abs(np.linalg.norm(obs-expected)/torso-row['error_to_nose_marker_gt_torso'])<1e-5
        hull=cv2.convexHull(pixels(surface(p[0],np.array([.22,.29,.21]),basis,128)).astype(np.float32))
        inside=cv2.pointPolygonTest(hull,tuple(obs),False)>=0
        assert inside==row['inside_projected_head']
        if row['image'] in result['example_images']:
            _,_,state=capture(pose.joints,cam)
            assert int((state['surface_ids']==24).sum())==row['visible_nose_pixels'];replayed+=1
    for g in ('all',*sorted({r['group'] for r in result['records']})):
        rows=result['records'] if g=='all' else [r for r in result['records'] if r['group']==g]
        assert summary(rows)==result['summary'][g]
    save_json(args.out/'audit.json',{'input_images':360,'independent_projection_checks':360,
        'maximum_nose_pixel_error':maxerr,'example_visibility_replays':replayed,
        'head_region_membership_recomputed':360,'summary_recomputed':True,
        'source_hashes_unchanged':True,'script_sha256':sha256(Path(__file__)),
        'results_sha256':sha256(args.out/'results.json')})
    print('PASS360 independent locations/regions; max pixel error',maxerr,'visibility replays',replayed)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('out','library-root'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
