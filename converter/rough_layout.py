"""Fit final, immutable people to source-image observations. No pose editing.

Depth is an apparent-size estimate under a shared 40 degree camera, not a
measurement of real-world distance. Expected source occlusion is preserved.
"""
import math

import numpy as np

VERSION = "rough-layout-v1"


def camera_for(width, height):
    return dict(width=width, height=height, horizontal_fov_degrees=40.0,
                focal_pixels=width / (2 * math.tan(math.radians(20))),
                principal_pixels=[width / 2, height / 2],
                position_blender=[0.0, -10.0, 1.5], projection="perspective")


def project(points, translation, camera):
    p = np.asarray(points, float) + np.asarray(translation, float)
    depth = p[:, 1] - camera["position_blender"][1]
    if np.min(depth) <= 0.05:
        raise ValueError("rough layout puts skin behind the camera")
    f = camera["focal_pixels"]
    cx, cy = camera["principal_pixels"]
    return np.column_stack((cx + f * p[:, 0] / depth,
                            cy - f * (p[:, 2] - camera["position_blender"][2]) / depth))


def fit_translation(vertices, joints, observation, camera):
    """Match observed head position and visible height under the shared camera.

    Only root translation changes. Scale, rotations, fingers and skin topology
    stay exactly as reviewed, including partial-scope exports.
    """
    vertices = np.asarray(vertices, float)
    target = np.asarray(observation["box"], float)
    center = (target[:2] + target[2:]) / 2
    points = np.asarray(observation["keypoints"], float)
    scores = np.asarray(observation["scores"], float)
    used = [i for i in range(5) if scores[i] >= .45
            and 0 <= points[i, 0] <= camera["width"] and 0 <= points[i, 1] <= camera["height"]]
    f = camera["focal_pixels"]
    cx, cy = camera["principal_pixels"]
    cam_y, cam_z = camera["position_blender"][1:]
    method = "head-anchor-visible-height" if used and -1 in joints else "visible-skin-box"
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    local = np.asarray(joints[-1]) if method.startswith("head") else (lo + hi) / 2
    anchor = np.average(points[used], axis=0, weights=scores[used]) if method.startswith("head") else center
    depth = float(np.clip(f * (hi[2] - lo[2]) / (target[3] - target[1]), .4 + np.ptp(vertices[:, 1]), 100))
    translation = np.array([(anchor[0] - cx) * depth / f - local[0],
                            cam_y + depth - local[1],
                            cam_z + (cy - anchor[1]) * depth / f - local[2]])
    def residual(t):
        xy = project(vertices, t, camera)
        actual_anchor = project([local], t, camera)[0] if method.startswith("head") else (xy.min(0) + xy.max(0)) / 2
        return np.r_[actual_anchor - anchor, np.ptp(xy[:, 1]) - (target[3] - target[1])]
    for _ in range(50):
        r = residual(translation)
        if np.max(np.abs(r)) < .05:
            break
        eps = 1e-4
        jac = np.column_stack([(residual(translation + np.eye(3)[i] * eps) - r) / eps for i in range(3)])
        step = np.linalg.lstsq(jac, -r, rcond=None)[0]
        accepted = False
        for factor in (1, .5, .25, .125, .0625):
            candidate = translation + step * factor
            if not np.isfinite(candidate).all() or np.min(vertices[:, 1] + candidate[1] - cam_y) <= .1:
                continue
            if np.linalg.norm(residual(candidate)) < np.linalg.norm(r):
                translation, accepted = candidate, True
                break
        if not accepted:
            break
    if np.max(np.abs(residual(translation))) > 2:
        raise ValueError("rough layout could not match observed anchors and visible size")
    if np.max(np.abs(translation)) > 1000 or np.median(vertices[:, 1]) + translation[1] - cam_y > 100:
        raise ValueError("rough layout exceeds supported scene extent")
    xy = project(vertices, translation, camera)
    box = np.r_[xy.min(axis=0), xy.max(axis=0)]
    return translation.tolist(), dict(method=method, observed_box=target.tolist(),
        projected_skin_box=box.tolist(), face_joint_indices=used,
        anchor_error_pixels=float(np.linalg.norm(residual(translation)[:2])),
        visible_height_error_pixels=float(abs(residual(translation)[2])),
        physical_scale=1.0, additional_separation_pixels=0,
        estimated_camera_depth_m=float(np.median(vertices[:, 1]) + translation[1] - cam_y))


def geometry_for(rig, meshes):
    import bpy
    graph = bpy.context.evaluated_depsgraph_get()
    vertices, head = [], []
    for obj in meshes:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            vertices.extend(tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices)
            groups = {g.index for g in obj.vertex_groups if g.name.split(":")[-1] == "Head"}
            head.extend(tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices
                        if sum(g.weight for g in v.groups if g.group in groups) >= .4)
        finally:
            evaluated.to_mesh_clear()
    names = {5: "LeftArm", 6: "RightArm", 11: "LeftUpLeg", 12: "RightUpLeg"}
    joints = {}
    if head:
        joints[-1] = np.mean(head, axis=0).tolist()
    for index, suffix in names.items():
        matches = [b for b in rig.data.bones if b.name.split(":")[-1] == suffix]
        if len(matches) == 1:
            joints[index] = tuple(rig.matrix_world @ matches[0].head_local)
    return vertices, joints
