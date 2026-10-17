"""Partial contour comparison against hash-pinned meshes at one fixed pose/view.

Diagnostic only: this experiment cannot change the BFF body/default selection.
The camera is shared by all 20 bodies. No hidden anatomy or candidate-specific warp.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .rough_evidence import PARTS, SAMPLES, segment, torso_length

VERSION = "rough-projection-match.v1"
MIN_SAMPLES = 4
MIN_PARTS = 3
MIN_REGIONS = 2
MAX_MEDIAN_POSE_ANGLE = 20.0
MAX_POSE_ANGLE = 30.0
MIN_MARGIN = 0.025


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def part_labels(image, colors_linear):
    """Decode opaque, unlit, no-AA material IDs; unknown colors fail closed."""
    rgb = np.asarray(image.convert("RGBA"))
    palette = np.asarray(colors_linear, float)
    srgb = np.where(
        palette <= 0.0031308, 12.92 * palette, 1.055 * palette ** (1 / 2.4) - 0.055
    )
    palette = np.rint(srgb * 255)
    d = ((rgb[:, :, None, :3].astype(float) - palette[None, None]) ** 2).sum(-1)
    nearest = d.argmin(-1)
    labels = np.where((rgb[:, :, 3] > 250) & (d.min(-1) <= 27), nearest + 1, 0)
    # background=0, unmeasured head/hands=1, torso=2, ...
    return labels.astype("uint8")


def projection_profiles(record):
    from PIL import Image

    with Image.open(record["image"]) as im:
        labels = part_labels(im, record["colors_linear"])
    points, mask = record["keypoints"], record["valid_joint_mask"]
    scale = torso_length(points, mask)
    if not scale:
        raise ValueError("reference_torso_missing")
    h, w = labels.shape
    result = {}
    for label, name in enumerate(PARTS, 2):
        s = segment(points, mask, name)
        if s is None:
            continue
        a, b = s
        normal = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        samples = {}
        offsets = np.arange(-int(scale), int(scale) + 1, dtype=float)
        for t in SAMPLES:
            xy = a + (b - a) * t + offsets[:, None] * normal
            ij = np.rint(xy).astype(int)
            valid = (
                (ij[:, 0] > 0)
                & (ij[:, 0] < w - 1)
                & (ij[:, 1] > 0)
                & (ij[:, 1] < h - 1)
            )
            values = np.zeros(len(offsets), int)
            values[valid] = labels[ij[valid, 1], ij[valid, 0]]
            bounds = np.flatnonzero(np.diff(np.r_[False, values == label, False]))
            runs = [
                (l, r)
                for l, r in zip(bounds[::2], bounds[1::2])
                if offsets[l] <= 0 <= offsets[r - 1]
            ]
            if len(runs) != 1:
                continue
            l, r = runs[0]
            # Both boundaries must border background. Another body part is occlusion.
            if (
                l < 1
                or r >= len(values)
                or not valid[[l - 1, r]].all()
                or values[l - 1] != 0
                or values[r] != 0
            ):
                continue
            width = (offsets[r - 1] - offsets[l]) / scale
            if width <= 0:
                continue
            samples[round(float(t), 4)] = float(width)
        result[name] = samples
    return result


def verify_projection_group(records, assets, job):
    expected = {a["body_id"]: a["asset_sha256"] for a in assets}
    if len(expected) != len(assets) or not expected:
        raise ValueError("invalid_asset_catalog")
    ids = [r["body_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("incomplete_projection_catalog")
    shared = None
    for r in records:
        if r["asset_sha256"] != expected[r["body_id"]]:
            raise ValueError("projection_asset_mismatch")
        if any(r[k] != job[k] for k in ("key", "pose_id", "pose_sha256", "view")):
            raise ValueError("projection_pose_view_mismatch")
        if sha256(r["image"]) != r["image_sha256"]:
            raise ValueError("projection_pixels_mismatch")
        root = Path(__file__).resolve().parents[3]
        if r["renderer_sha256"] != sha256(
            root / "scripts/render_body_pose_projections.py"
        ) or r["retarget_sha256"] != sha256(root / "converter/retarget.py"):
            raise ValueError("projection_implementation_changed")
        if r["parts"] != list(PARTS) or not r["mesh_finite"]:
            raise ValueError("invalid_projection_parts")
        import json

        fingerprint = json.dumps(
            [
                r["camera"],
                r["renderer_sha256"],
                r["retarget_sha256"],
                r["colors_linear"],
            ],
            sort_keys=True,
        )
        if shared is not None and fingerprint != shared:
            raise ValueError("candidate_specific_projection")
        shared = fingerprint
        p = np.asarray(r["keypoints"], float)
        m = np.asarray(r["valid_joint_mask"], bool)
        if p.shape != (17, 2) or m.shape != (17,) or not np.isfinite(p[m]).all():
            raise ValueError("invalid_projection_pose")
    if sha256(job["bvh_path"]) != job["pose_sha256"]:
        raise ValueError("source_pose_changed")


def pose_compatibility(points, mask, reference, parts):
    """Direction-only pose gate; no penalty for the proportions being estimated."""
    angles = {}
    for part in parts:
        q = segment(points, mask, part)
        r = segment(reference["keypoints"], reference["valid_joint_mask"], part)
        if q is None or r is None:
            continue
        u = q[1] - q[0]
        v = r[1] - r[0]
        angles[part] = float(
            np.degrees(
                np.arccos(
                    np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)
                )
            )
        )
    passed = bool(
        len(angles) >= 2
        and np.median(list(angles.values())) <= MAX_MEDIAN_POSE_ANGLE
        and max(angles.values()) <= MAX_POSE_ANGLE
    )
    return dict(
        passed=passed,
        angles_degrees=angles,
        median_degrees=float(np.median(list(angles.values()))) if angles else None,
    )


def rank_profiles(observed, references, torso_pixels):
    """Identical parts/stations and equal part weights for EVERY candidate."""
    if not np.isfinite(torso_pixels) or torso_pixels <= 4 or len(references) < 2:
        raise ValueError("invalid_comparison_inputs")
    common = {}
    for part, row in observed.items():
        if not row.get("accepted"):
            continue
        query = {s["t"]: s["width"] for s in row["samples"]}
        if len(query) != len(row["samples"]) or any(
            not np.isfinite(v) or v <= 0 for v in query.values()
        ):
            raise ValueError("invalid_observed_profile")
        keys = set(query)
        for candidate in references.values():
            keys &= set(candidate.get(part, {}))
        if len(keys) >= MIN_SAMPLES:
            common[part] = sorted(keys)
    if not common:
        return dict(
            ranking=[], common_parts=[], reason_codes=["no_shared_visible_profiles"]
        )
    distances = {}
    for body, profile in references.items():
        per = {
            p: float(
                np.mean(
                    [
                        abs(
                            profile[p][t]
                            - next(
                                s["width"]
                                for s in observed[p]["samples"]
                                if s["t"] == t
                            )
                        )
                        for t in ts
                    ]
                )
            )
            for p, ts in common.items()
        }
        distances[body] = per
    ranking = sorted(
        [
            dict(body_id=b, distance=float(np.mean(list(d.values()))), part_distances=d)
            for b, d in distances.items()
        ],
        key=lambda x: (x["distance"], x["body_id"]),
    )
    regions = {
        ("torso" if p == "torso" else "arms" if "arm" in p else "legs") for p in common
    }
    reasons = []
    if len(common) < MIN_PARTS or len(regions) < MIN_REGIONS:
        reasons.append("insufficient_independent_regions")
    margin = (
        ranking[1]["distance"] - ranking[0]["distance"] if len(ranking) > 1 else 0.0
    )
    uncertainty = max(MIN_MARGIN, 4.0 / torso_pixels)
    if margin <= uncertainty:
        reasons.append("margin_below_contour_uncertainty")
    winner = ranking[0]["body_id"]
    loo = []
    if len(common) >= 2:
        for omit in common:
            scores = sorted(
                (float(np.mean([v for k, v in d.items() if k != omit])), b)
                for b, d in distances.items()
            )
            loo.append(dict(omitted_part=omit, body_id=scores[0][1]))
        if any(x["body_id"] != winner for x in loo):
            reasons.append("part_ablation_changes_winner")
    return dict(
        ranking=ranking,
        common_parts=list(common),
        sample_counts={p: len(v) for p, v in common.items()},
        regions=sorted(regions),
        margin=margin,
        uncertainty_floor=uncertainty,
        leave_one_part_out=loo,
        reason_codes=reasons,
    )


def compare_rough(case, records, assets, jobs):
    """Return diagnostics and abstention; never a production body selection."""
    out = dict(
        version=VERSION,
        case_id=case["case_id"],
        status="held",
        auto_apply=False,
        selected_body_id=None,
        hypotheses=[],
        reason_codes=[],
    )
    evidence = case["segmentation"]
    if not case["ownership_verified"] or evidence.get("reason_codes"):
        out["reason_codes"] = ["ownership_or_scale_unverified"]
        return out
    if not evidence["accepted_parts"]:
        out["reason_codes"] = ["no_image_supported_contours"]
        return out
    if not jobs:
        out["reason_codes"] = ["no_fixed_pose_projections"]
        return out
    for job in jobs:
        group = [r for r in records if r["key"] == job["key"]]
        try:
            verify_projection_group(group, assets, job)
            # A single fixed neutral body defines pose agreement for all candidates.
            anchor = next(r for r in group if r["body_id"] == assets[0]["body_id"])
            fit = pose_compatibility(
                case["points"], case["mask"], anchor, evidence["accepted_parts"]
            )
            profiles = {r["body_id"]: projection_profiles(r) for r in group}
            comparison = rank_profiles(
                evidence["parts"], profiles, evidence["torso_pixels"]
            )
            reasons = list(comparison["reason_codes"])
            if not fit["passed"]:
                reasons.insert(0, "pose_view_mismatch")
            out["hypotheses"].append(
                dict(
                    key=job["key"],
                    pose_id=job["pose_id"],
                    view=job["view"],
                    pose_fit=fit,
                    comparison=comparison,
                    reason_codes=reasons,
                )
            )
        except (ValueError, KeyError, OSError, StopIteration) as exc:
            out["hypotheses"].append(
                dict(key=job["key"], reason_codes=["projection_invalid:" + str(exc)])
            )
    winners = {
        h["comparison"]["ranking"][0]["body_id"]
        for h in out["hypotheses"]
        if h.get("comparison", {}).get("ranking")
    }
    if len(winners) > 1:
        out["reason_codes"].append("pose_hypotheses_disagree")
    eligible = [h for h in out["hypotheses"] if not h["reason_codes"]]
    if (
        len(eligible) == len(jobs)
        and len({h["comparison"]["ranking"][0]["body_id"] for h in eligible}) == 1
    ):
        out["diagnostic_body_id"] = eligible[0]["comparison"]["ranking"][0]["body_id"]
        out["status"] = "review_candidate"
    elif eligible:
        out["reason_codes"].append("pose_hypotheses_disagree_or_incomplete")
    out["reason_codes"] += sorted(
        {r for h in out["hypotheses"] for r in h["reason_codes"]}
    )
    # Neither the retargeting QA nor rough matching thresholds are production-calibrated.
    out["reason_codes"].append("experimental_unvalidated_no_auto_apply")
    return out
