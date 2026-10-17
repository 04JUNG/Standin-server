"""Post-hoc rendering diagnosis; never changes retrieval or frozen renderer.

Capture renderer geometry read-only at return, then independently project a
dense ellipsoid surface. This tests the suspect head silhouette, not anatomy
or human relevance. Existing evaluated queries are not a new holdout.
"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose, save_json, sha256
from standin_eval.perspective_search import Camera, camera_points, project, normalize_batch


def surface(center, radii, basis, resolution=256):
    theta = np.linspace(0, np.pi, resolution + 1)
    phi = np.arange(resolution * 2) * np.pi / resolution
    s = np.sin(theta)[:, None]
    xyz = np.stack(np.broadcast_arrays(s * np.cos(phi),
                   s * np.sin(phi), np.cos(theta)[:, None]), axis=-1)
    return (xyz.reshape(-1, 3) * radii) @ basis.T + center


def plane(points, distance):
    if distance is None:
        return points[:, :2].copy()
    depth = points[:, 2] + distance
    if np.any(depth <= 0):
        raise ValueError('surface crosses camera')
    return points[:, :2] / depth[:, None]


def capture(joints, camera):
    if sys.getprofile() is not None:
        raise RuntimeError('do not replace an existing profiler')
    saved = {}
    def trace(frame, event, arg):
        if frame.f_code is render_pose.__code__ and event == 'return':
            for key in ('objects', 'surface_ids', 'scale', 'cx', 'cy', 'points'):
                if key in frame.f_locals:
                    saved[key] = frame.f_locals[key]
    sys.setprofile(trace)
    try:
        im, xy = render_pose(joints, camera.yaw, size=512,
                             pitch=camera.pitch, distance=camera.distance)
    finally:
        sys.setprofile(None)
    return im, xy, saved


def head_metrics(joints, camera, resolution=128):
    p = camera_points(joints, camera).astype(float)
    up = (p[5] + p[6]) / 2
    up /= np.linalg.norm(up)
    lateral = p[5] - p[6]
    lateral /= np.linalg.norm(lateral)
    forward = np.cross(up, lateral)
    forward /= np.linalg.norm(forward)
    lateral = np.cross(forward, up)
    basis = np.column_stack((lateral, up, forward))
    mesh = surface(p[0], np.array([.22, .29, .21]), basis, resolution)
    hp = plane(mesh, camera.distance)
    bp = plane(p[5:], camera.distance)
    span = np.ptp(hp, axis=0)
    result = {'head_projected_width': float(span[0]),
              'head_projected_height': float(span[1]),
              'head_height_width_ratio': float(span[1] / span[0]),
              'head_height_over_body_joint_span': float(span[1] / np.ptp(bp[:, 1]))}
    if camera.distance is not None:
        depth = p[:, 2] + camera.distance
        result.update(head_center_depth=float(depth[0]),
                      head_surface_min_depth=float(mesh[:, 2].min() + camera.distance),
                      head_off_axis_degrees=float(np.rad2deg(np.arctan2(
                          np.linalg.norm(p[0, :2]), depth[0]))),
                      body_depth_ratio=float(depth[5:].max()/depth[5:].min()))
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'out/refinement-real-roughs-20261013/rankings.json'
    frozen = {str(p.relative_to(ROOT)): sha256(p) for p in
              (source, ROOT/'scripts/experiment_z_rendered.py',
               ROOT/'standin_eval/camera_refinement.py')}
    save_json(args.out/'protocol.json', {
        'kind': 'post-hoc geometry diagnosis, no accuracy evaluation',
        'target': {'image': 'rough_315.png', 'rank': 5,
                   'pose_id': 'Jumping Down_00011_mirror'},
        'controlled_variable': 'distance only; yaw 191.25 and pitch -5.625 fixed',
        'distances': [1.1851851851851851, 2., 4., None],
        'independent_check': 'dense ellipsoid surface projection vs visible head pixels',
        'census': 'all 40 selected-arm real rough candidates; no threshold selection',
        'frozen_sources': frozen, 'script_sha256': sha256(Path(__file__))})
    poses = load_library(args.library_root)
    by_id = {p.pose_id: p for p in poses}
    joints = by_id['Jumping Down_00011_mirror'].joints
    sheet = Image.new('RGB', (2048, 640), 'white')
    pen = ImageDraw.Draw(sheet)
    results = []
    for col, distance in enumerate((1.1851851851851851, 2., 4., None)):
        camera = Camera(191.25, -5.625, distance)
        im, xy, state = capture(joints, camera)
        # Frozen renderer's object order: torso/pelvis, 16 limb/joints,
        # two hands, two feet, neck, head, nose and eyes.
        assert len(state['objects']) == 27
        center, radii, basis, _ = state['objects'][23]
        assert np.array_equal(radii, [.22, .29, .21])
        hp = plane(surface(center, radii, basis), distance)
        px = (hp - [state['cx'], state['cy']]) * [state['scale'], -state['scale']] + 256
        hull = cv2.convexHull(px.astype(np.float32)).reshape(-1, 2)
        mask = np.zeros((512, 512), np.uint8)
        cv2.fillConvexPoly(mask, np.rint(hull*256).astype(np.int32), 1, shift=8)
        visible = state['surface_ids'] == 23
        ys, xs = np.where(visible)
        distances = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True)
                              for x, y in zip(xs, ys)])
        expected, valid = project(joints, camera)
        error = float(np.max(np.abs(normalize_batch(xy)-normalize_batch(expected))))
        assert valid and error < 1e-5
        # A tessellated surface is approximate; tolerate <1 pixel inward edge.
        assert len(distances) and distances.min() > -1.
        metrics = head_metrics(joints, camera)
        metrics.update(distance=distance, head_visible_pixels=int(visible.sum()),
                       mesh_mask_iou=float((visible & (mask>0)).sum()/(visible | (mask>0)).sum()),
                       worst_visible_pixel_outside_mesh_px=float(max(0., -distances.min())),
                       normalized_joint_projection_max_error=error,
                       visible_head_bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())])
        overlay = ImageDraw.Draw(im)
        overlay.line([tuple(v) for v in hull]+[tuple(hull[0])], fill='#318bc2', width=2)
        for a,b in ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
                    (11,12),(11,13),(13,15),(12,14),(14,16)):
            overlay.line([tuple(xy[a]),tuple(xy[b])],fill='#da4c59',width=2)
        im.save(args.out/f'view-{col}.png')
        sheet.paste(im,(512*col,60))
        pen.text((512*col+15,15), f'd = {distance if distance else "orthographic"}',fill='black')
        pen.text((512*col+15,35), 'Same pose, yaw=191.25, pitch=-5.625',fill='black')
        pen.text((512*col+15,580), f'Head H/W: {metrics["head_height_width_ratio"]:.3f}',fill='black')
        pen.text((512*col+15,600), f'Head H / body joint span: {metrics["head_height_over_body_joint_span"]:.3f}',fill='black')
        results.append(metrics)
    sheet.save(args.out/'distance-comparison.png')
    census = []
    for row in json.loads(source.read_text()):
        for rank,candidate in enumerate(row.get('keepfirst_blend75',[]),1):
            camera = Camera(**candidate['camera'])
            census.append({'image':row['image'],'rank':rank,'pose_id':candidate['pose_id'],
                           'camera':candidate['camera'],
                           **head_metrics(by_id[candidate['pose_id']].joints,camera)})
    assert len(census)==40
    assert all(sha256(ROOT/p)==digest for p,digest in frozen.items())
    save_json(args.out/'results.json', {'views':results,'census':census,
              'source_hashes_unchanged':True,'accuracy':None,
              'limitations':['Surface check covers this head silhouette, not all renderer code.',
                             'Head frame follows torso; BVH head rotation is not represented.',
                             'No human relevance rating or new holdout.']})
    print(json.dumps(results, indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--library-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
