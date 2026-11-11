"""Retrieve from body-relative 3D relations and a second local 2D recognizer."""

import numpy as np
from .crop_joints import FLIP
from .crop_joints import restore_rotation
from .pose_recall_probe import observation
from .pose_recall_probe import ranks


ANCHORS = [5, 6, 11, 12]


EDGES = np.asarray([(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)])


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / norms.clip(1e-6), norms[..., 0]


def body_frame(points):
    hip = points[..., [11, 12], :].mean(axis=-2)
    shoulders = points[..., [5, 6], :].mean(axis=-2)
    up, scale = normalize(shoulders - hip)
    side = (points[..., 6, :] - points[..., 5, :] + points[..., 12, :] - points[..., 11, :]) * .5
    side -= (side * up).sum(axis=-1, keepdims=True) * up
    right, width = normalize(side)
    forward = np.cross(right, up)
    frame = np.stack((right, up, forward), axis=-1)
    relative = (points - hip[..., None, :]) / scale[..., None, None].clip(1e-6)
    return relative @ frame, (scale > 1e-6) & (width > .16 * scale)


class BodyFrameIndex:
    def __init__(self, camera):
        self.gallery, self.valid = body_frame(camera.points)
        self.scores = camera.scores
        self.valid &= (self.scores[:, ANCHORS] >= .3).all(axis=1)
        self.edges, _ = normalize(self.gallery[:, EDGES[:, 1]] - self.gallery[:, EDGES[:, 0]])

    def costs(self, model, z, lower):
        xy = model['keypoints']
        confidence, visible = model['joint_scores'], model['joint_mask']
        mask = visible & (confidence >= .38)
        if not mask[ANCHORS].all() or mask[5:].sum() < 7:
            return None
        if not lower:
            mask[13:] = False
        use = np.flatnonzero(mask[5:]) + 5
        edges = mask[EDGES].all(axis=1)
        if edges.sum() < 2:
            return None
        torso_2d = np.linalg.norm(xy[[5, 6]].mean(0) - xy[[11, 12]].mean(0))
        if torso_2d < 8:
            return None
        xy = (xy - xy[[11, 12]].mean(0)) / torso_2d
        depth = np.clip((z - z[[11, 12]].mean()) / .5, -2., 2.)
        weights = confidence[use] / confidence[use].sum()
        edge_weights = confidence[EDGES[edges]].min(axis=1)
        edge_weights /= edge_weights.sum()
        options = []
        for depth_scale in (.55, 1., 1.55):
            query = np.column_stack((xy[:, 0], -xy[:, 1], depth * depth_scale))
            query, usable = body_frame(query)
            if not usable:
                continue
            directions, _ = normalize(query[EDGES[edges, 1]] - query[EDGES[edges, 0]])
            for sign in (1., -1.):
                orient = np.array([1., 1., sign])
                residual = np.linalg.norm(self.gallery[:, use] - query[use] * orient, axis=-1)
                angle = (1. - np.clip(np.einsum('ned,ed->ne', self.edges[:, edges],
                                                directions * orient), -1, 1)) * .5
                cost = .70 * (np.minimum(residual, 1.5) @ weights) + .30 * (angle @ edge_weights)
                options.append(cost + .027 * abs(np.log(depth_scale)))
        if not options:
            return None
        valid = self.valid & (self.scores[:, use] >= .3).all(axis=1)
        return np.where(valid, np.min(options, axis=0), np.inf)


def box_hypotheses(model, image, box, lower):
    if not box:
        return
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(box)
    dx, dy = .12 * (x2 - x1), .12 * (y2 - y1)
    a, b, c, d = np.rint(np.clip([x1-dx, y1-dy, x2+dx, y2+dy],
                                [0, 0, 0, 0], [w, h, w, h])).astype(int)
    if min(c - a, d - b) < 28:
        return
    crop = image[b:d, a:c]
    ch, cw = crop.shape[:2]
    best = []
    for turn in range(4):
        view = np.ascontiguousarray(np.rot90(crop, turn))
        vh, vw = view.shape[:2]
        points, scores = model(view[..., ::-1].copy(), bboxes=[[0, 0, vw, vh]])
        coords = restore_rotation(points[0, :17], turn, cw, ch) + [a, b]
        value = observation(coords, scores[0, :17], (h, w), lower)
        if value is not None:
            quality = np.mean(np.where(value['joint_mask'][5:], value['joint_scores'][5:], 0.))
            best.append((quality, turn, value))
        if turn == 0:
            other, other_scores = model(view[:, ::-1, ::-1].copy(), bboxes=[[0, 0, vw, vh]])
            mirror = other[0, FLIP].copy()
            mirror[:, 0] = vw - 1 - mirror[:, 0]
            agree = np.linalg.norm(points[0, :17] - mirror, axis=-1) <= .13 * max(ch, cw)
            fused = observation((points[0, :17] + mirror) * .5 + [a, b],
                                np.minimum(scores[0, :17], other_scores[0, FLIP]), (h, w), lower, mask=agree)
            if fused is not None:
                yield 'body_x_mirror', fused
    if best:
        yield 'body_x', max(best, key=lambda result: result[0])[2]


def profiles(ids, costs, semantic, golds):
    result = {'geometry': ranks(ids, costs, golds)}
    if semantic is not None:
        result['hybrid'] = ranks(ids, costs - .24 * semantic, golds)
    return result
