"""Optional local SAM 2 adapter and image-supported partial body masks.

SAM confidence is never a body-match probability. A segmentation boundary must
also be supported by original ink edges; occluded parts remain excluded.
"""

from __future__ import annotations
import hashlib
import numpy as np
from .rough_evidence import PARTS, SAMPLES, segment, torso_length, edge_map

VERSION = "rough-body-segmentation.v2"


def checkpoint_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class Sam2PartSegmenter:
    def __init__(self, checkpoint, expected_sha256, *, device="cpu"):
        if checkpoint_sha256(checkpoint) != expected_sha256:
            raise ValueError("segmentation_checkpoint_mismatch")
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        torch.set_num_threads(4)
        self.predictor = SAM2ImagePredictor(
            build_sam2(
                "configs/sam2.1/sam2.1_hiera_t.yaml",
                str(checkpoint),
                device=device,
                apply_postprocessing=False,
            )
        )
        self.checkpoint_sha256 = expected_sha256

    def segment(self, image, points, mask, visibility, other_points=()):
        import torch

        p = np.asarray(points, float)
        m = np.asarray(mask, bool)
        scale = torso_length(p, m)
        w, h = image.size
        if not scale:
            raise ValueError("segmentation_requires_torso")
        output = {}
        whole = None
        with torch.inference_mode():
            self.predictor.set_image(np.array(image.convert("RGB"), copy=True))
            centers = {
                k: (s[0] + s[1]) / 2
                for k in PARTS
                if (s := segment(p, m, k)) is not None
            }
            for part in PARTS:
                if (
                    visibility["parts"][part]["state"] not in {"exposed", "fitted"}
                    or part not in centers
                ):
                    continue
                a, b = segment(p, m, part)
                positive = np.array([a + (b - a) * t for t in (0.3, 0.5, 0.7)])
                if not (
                    (positive[:, 0] >= 0)
                    & (positive[:, 0] < w)
                    & (positive[:, 1] >= 0)
                    & (positive[:, 1] < h)
                ).all():
                    continue
                pad = scale * (0.55 if part == "torso" else 0.3)
                box = np.r_[
                    np.maximum(np.minimum(a, b) - pad, [1, 1]),
                    np.minimum(np.maximum(a, b) + pad, [w - 2, h - 2]),
                ]
                # Adjacent body centers are not reliable negative pixels in overlapping poses.
                negative = list(other_points)
                negative = [v for v in negative if 0 <= v[0] < w and 0 <= v[1] < h]
                coords = np.concatenate([positive, np.asarray(negative).reshape(-1, 2)])
                labels = np.r_[
                    np.ones(len(positive), int), np.zeros(len(negative), int)
                ]
                masks, scores, logits = self.predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    box=box,
                    multimask_output=True,
                )
                idx = int(np.argmax(scores))
                output[part] = dict(
                    mask=masks[idx].astype(bool),
                    predicted_iou=float(scores[idx]),
                    prompt_box=box.tolist(),
                )
            # Person silhouette for review, not completion of hidden anatomy.
            positive = np.asarray(
                [v for v in centers.values() if 0 <= v[0] < w and 0 <= v[1] < h]
            ).reshape(-1, 2)
            negative = np.asarray(
                [v for v in other_points if 0 <= v[0] < w and 0 <= v[1] < h]
            ).reshape(-1, 2)
            if len(positive):
                coords = np.concatenate([positive, negative])
                labels = np.r_[
                    np.ones(len(positive), int), np.zeros(len(negative), int)
                ]
                masks, scores, _ = self.predictor.predict(
                    point_coords=coords, point_labels=labels, multimask_output=True
                )
                idx = int(np.argmax(scores))
                whole = dict(
                    mask=masks[idx].astype(bool), predicted_iou=float(scores[idx])
                )
        return output, whole


def extract_mask_contours(
    image, points, mask, visibility, segments, *, ownership_verified, other_points=()
):
    import cv2

    w, h = image.size
    scale = torso_length(points, mask)
    edges, removed = edge_map(image)
    distance = cv2.distanceTransform((~edges).astype("uint8"), cv2.DIST_L2, 3)
    observed = np.zeros((h, w), np.uint8)
    excluded = np.zeros((h, w), np.uint8)
    output = dict(
        version=VERSION,
        torso_pixels=scale,
        parts={},
        accepted_parts=[],
        ownership_verified=bool(ownership_verified),
        reason_codes=[],
    )
    if not scale or not ownership_verified or visibility.get("ownership_ambiguous"):
        output["reason_codes"] = ["ownership_or_scale_unverified"]
        return output, observed, excluded
    for name in PARTS:
        v = visibility["parts"][name]
        s = segment(points, mask, name)
        row = dict(
            state=v["state"],
            evidence=v["evidence"],
            samples=[],
            accepted=False,
            reason=None,
        )
        output["parts"][name] = row
        if s is None:
            row["reason"] = "missing_or_degenerate_pose_segment"
            continue
        a, b = s
        direction = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-direction[1], direction[0]])
        if v["state"] not in {"exposed", "fitted"}:
            row["reason"] = "part_" + v["state"]
            if v["state"] in {"loose_clothing", "occluded"}:
                cv2.line(
                    excluded,
                    tuple(np.rint(a).astype(int)),
                    tuple(np.rint(b).astype(int)),
                    255,
                    max(2, int(scale * 0.25)),
                )
            continue
        seg = segments.get(name)
        if not seg:
            row["reason"] = "segmentation_missing"
            continue
        sm = np.asarray(seg["mask"], bool)
        if sm.shape != (h, w):
            raise ValueError("segmentation_shape_mismatch")
        row["mask_area"] = int(sm.sum())
        row["predicted_iou"] = float(seg["predicted_iou"])
        if not np.isfinite(row["predicted_iou"]) or row["predicted_iou"] < 0.70:
            row["reason"] = "segmentation_low_quality"
            continue
        radius = scale * (0.7 if name == "torso" else 0.4)
        offsets = np.arange(-int(radius), int(radius) + 1, dtype=float)
        for t in SAMPLES:
            center = a + (b - a) * t
            xy = center + offsets[:, None] * normal
            ij = np.rint(xy).astype(int)
            valid = (
                (ij[:, 0] > 1)
                & (ij[:, 0] < w - 2)
                & (ij[:, 1] > 1)
                & (ij[:, 1] < h - 2)
            )
            inside = np.zeros(len(offsets), bool)
            inside[valid] = sm[ij[valid, 1], ij[valid, 0]]
            bounds = np.flatnonzero(np.diff(np.r_[False, inside, False]))
            runs = list(zip(bounds[::2], bounds[1::2]))
            # The predicted region must contain the detector's segment axis.
            runs = [(lo, hi) for lo, hi in runs if offsets[lo] <= 0 <= offsets[hi - 1]]
            if len(runs) != 1:
                continue
            lo, hi = runs[0]
            lo += 1
            hi -= 2
            if (
                hi <= lo
                or lo <= 1
                or hi >= len(offsets) - 2
                or not valid[[lo, hi]].all()
            ):
                continue
            lp, rp = ij[lo], ij[hi]
            width = (offsets[hi] - offsets[lo]) / scale
            if not (0.06 <= width <= (1.15 if name == "torso" else 0.65)):
                continue
            edge_error = float(max(distance[lp[1], lp[0]], distance[rp[1], rp[0]]))
            if edge_error > max(2.5, scale * 0.015):
                continue
            if any(
                min(
                    np.linalg.norm(lp - np.asarray(z)),
                    np.linalg.norm(rp - np.asarray(z)),
                )
                < scale * 0.08
                for z in other_points
            ):
                continue
            row["samples"].append(
                dict(
                    t=round(float(t), 4),
                    left=xy[lo].tolist(),
                    right=xy[hi].tolist(),
                    width=float(width),
                    edge_error_pixels=edge_error,
                )
            )
        row["boundary_support"] = len(row["samples"]) / len(SAMPLES)
        if len(row["samples"]) < 4:
            row["reason"] = "insufficient_image_supported_boundaries"
            continue
        widths = np.array([x["width"] for x in row["samples"]])
        if np.median(abs(np.diff(widths))) > max(0.04, float(np.median(widths)) * 0.3):
            row["reason"] = "inconsistent_boundary_track"
            continue
        row["accepted"] = True
        output["accepted_parts"].append(name)
        for sample in row["samples"]:
            cv2.line(
                observed,
                tuple(np.rint(sample["left"]).astype(int)),
                tuple(np.rint(sample["right"]).astype(int)),
                255,
                2,
            )
    output["removed_overlay_pixels"] = int(removed.sum())
    return output, observed, excluded
