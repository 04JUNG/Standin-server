"""Preserve source head anchors/depth order; move only silhouette conflicts."""

from pathlib import Path
import json, math
import numpy as np
from .layout_math import project, box, fit_head, silhouette, overlaps, separate_masks


def rotate(points, yaw):
    a = math.radians(yaw)
    r = np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]]
    )
    return points @ r.T


def run(ctx):
    ROOT = ctx.work
    LAYOUT = BASE = ctx.baseline
    summaries = []
    for item in json.loads((ROOT / "inputs.json").read_text()):
        cut = item["cut_id"]
        folder = ROOT / cut
        old = json.loads((LAYOUT / cut / "layout.json").read_text())
        cfg = old["camera"]
        size = np.array([cfg["width"], cfg["height"]])
        actors = json.loads((folder / "geometry.json").read_text())["people"]
        oldrows = {p["person_index"]: p for p in old["people"]}
        sources = {
            p["person_index"]: p
            for p in json.loads((BASE / cut / "people.json").read_text())["people"]
        }
        geometries = []
        targets = []
        translations = []
        basis = []
        depths = []
        selections = {
            p["person_index"]: p
            for p in json.loads((folder / "selection.json").read_text())["people"]
        }
        for actor in actors:
            i = actor["person_index"]
            oldp = oldrows[i]
            g = np.load(folder / f"person_{i}-geometry.npz")
            g = dict(
                points=rotate(g["points"], actor["rotation_y_degrees"]),
                head_points=rotate(g["head_points"], actor["rotation_y_degrees"]),
                triangles=g["triangles"],
            )
            geometries.append(g)
            target = np.array(selections[i]["head_anchor_bbox"])
            targets.append(target)
            scale_points = g["head_points"]
            scale_target = target
            mode = selections[i]["head_scale_reason"]
            tr, _ = fit_head(
                g["head_points"],
                target,
                cfg,
                scale_points=scale_points,
                scale_target=scale_target,
            )
            translations.append(tr)
            basis.append(mode)
            depths.append(
                float(
                    g["head_points"][:, 1].mean() + tr[1] - cfg["position_blender"][1]
                )
            )
        # Keep the established front/back order while staying closest to the size fit.
        size_fit_depths = list(depths)
        order = sorted(
            range(len(actors)),
            key=lambda j: oldrows[actors[j]["person_index"]]["depth_m"],
        )
        pools = []
        for j in order:
            pools.append(([j], math.log(depths[j])))
            while len(pools) > 1 and pools[-2][1] > pools[-1][1]:
                ib, vb = pools.pop()
                ia, va = pools.pop()
                pools.append(
                    (ia + ib, (va * len(ia) + vb * len(ib)) / (len(ia) + len(ib)))
                )
        for members, value in pools:
            for j in members:
                constrained = math.exp(value)
                if abs(constrained - depths[j]) > 1e-9:
                    translations[j], _ = fit_head(
                        geometries[j]["head_points"],
                        targets[j],
                        cfg,
                        depth=constrained,
                        initial=translations[j],
                    )
                depths[j] = constrained
        translations = np.array(translations)
        rawtranslations = translations.copy()
        scale = 1200 / max(size)
        rawmasks = [
            silhouette(g["points"], g["triangles"], tr, cfg, scale)
            for g, tr in zip(geometries, translations)
        ]
        masks = rawmasks
        history = []
        total = np.zeros((len(actors), 2))
        for iteration in range(12):
            pairs = overlaps(masks)
            if not sum(p["pixels"] for p in pairs):
                break
            centers = np.array(
                [
                    (
                        box(project(g["head_points"], tr, cfg))[:2]
                        + box(project(g["head_points"], tr, cfg))[2:]
                    )
                    / 2
                    for g, tr in zip(geometries, translations)
                ]
            )
            shift, diag = separate_masks(
                masks,
                centers * scale,
                accumulated=total * scale if iteration < 2 else None,
                clearance=1 + iteration // 3,
            )
            history.append(
                dict(
                    iteration=iteration,
                    overlap_pixels=sum(p["pixels"] for p in pairs),
                    **diag,
                )
            )
            if not diag["success"]:
                break
            for j, (g, s) in enumerate(zip(geometries, shift / scale)):
                target = box(project(g["head_points"], translations[j], cfg)) + np.tile(
                    s, 2
                )
                translations[j], _ = fit_head(
                    g["head_points"],
                    target,
                    cfg,
                    depth=depths[j],
                    initial=translations[j],
                )
            total += shift / scale
            masks = [
                silhouette(g["points"], g["triangles"], tr, cfg, scale)
                for g, tr in zip(geometries, translations)
            ]
        rows = []
        for j, (actor, g, tr, target) in enumerate(
            zip(actors, geometries, translations, targets)
        ):
            i = actor["person_index"]
            oldp = oldrows[i]
            raw = box(project(g["head_points"], rawtranslations[j], cfg))
            final = box(project(g["head_points"], tr, cfg))
            center = (final[:2] + final[2:]) / 2
            prior = (target[:2] + target[2:]) / 2
            ray = -math.degrees(
                math.atan(
                    (center[0] - cfg["principal_pixels"][0]) / cfg["focal_pixels"]
                )
            )
            actualyaw = (actor["rotation_y_degrees"] - ray + 180) % 360 - 180
            depth = float(
                g["head_points"][:, 1].mean() + tr[1] - cfg["position_blender"][1]
            )
            assert abs(depth - depths[j]) < 1e-7
            rows.append(
                dict(
                    **actor,
                    target_head_bbox=target.tolist(),
                    raw_head_bbox=raw.tolist(),
                    final_head_bbox=final.tolist(),
                    raw_body_bbox=box(
                        project(g["points"], rawtranslations[j], cfg)
                    ).tolist(),
                    final_body_bbox=box(project(g["points"], tr, cfg)).tolist(),
                    target_evidence={"screen_size_basis": basis[j]},
                    raw_translation_blender=rawtranslations[j].tolist(),
                    translation_blender=tr.tolist(),
                    raw_position_three=[
                        float(rawtranslations[j, 0]),
                        float(rawtranslations[j, 2]),
                        float(-rawtranslations[j, 1]),
                    ],
                    position_three=[float(tr[0]), float(tr[2]), float(-tr[1])],
                    depth_m=depth,
                    previous_depth_m=oldp["depth_m"],
                    depth_change_ratio=depth / oldp["depth_m"],
                    shift_pixels=(center - prior).tolist(),
                    shift_frame=float(np.linalg.norm((center - prior) / size)),
                    head_height_ratio_to_previous=float(
                        (final[3] - final[1]) / (target[3] - target[1])
                    ),
                    actual_torso_camera_yaw=actualyaw,
                    torso_camera_yaw_drift_degrees=float(
                        (actualyaw - actor["torso_camera_yaw"] + 180) % 360 - 180
                    ),
                    scale=1.0,
                )
            )
        finalpairs = overlaps(masks)
        rawpairs = overlaps(rawmasks)
        depthorder = []
        for a in rows:
            for b in rows:
                if b["person_index"] <= a["person_index"]:
                    continue
                before = a["previous_depth_m"] - b["previous_depth_m"]
                after = a["depth_m"] - b["depth_m"]
                depthorder.append(
                    dict(
                        a=a["person_index"],
                        b=b["person_index"],
                        reversed=bool(before * after < -1e-7),
                        before_difference_m=before,
                        after_difference_m=after,
                    )
                )
        report = dict(
            cut_id=cut,
            camera=cfg,
            image_path=item["image_path"],
            image_sha256=item["image_sha256"],
            people=rows,
            raw_overlap_pairs=rawpairs,
            final_overlap_pairs=finalpairs,
            solver_history=history,
            zero_visible_silhouette_overlap=not any(p["pixels"] for p in finalpairs),
            depth_order=depthorder,
            size_fit_depths_before_order_constraint=size_fit_depths,
            constrained_depths=depths,
            manual_coordinates=False,
            manual_pose_changes=False,
            new_fresh_analysis=True,
        )
        (folder / "layout.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        (folder / "camera.json").write_text(json.dumps(cfg, indent=2))
        for variant, key in [
            ("raw", "raw_position_three"),
            ("final", "position_three"),
        ]:
            manifest = dict(
                schema_version=1,
                cut_id=cut,
                coordinate_system="threejs-y-up-m",
                persons=[
                    dict(
                        person_index=p["person_index"],
                        fbx_sha256=p["fbx_sha256"],
                        position=p[key],
                        rotation_y_degrees=p["rotation_y_degrees"],
                        scale=1.0,
                    )
                    for p in rows
                ],
            )
            (folder / f"{variant}-scene-input.json").write_text(
                json.dumps(manifest, indent=2)
            )
        summaries.append(
            dict(
                cut_id=cut,
                people=len(rows),
                raw_overlap_pixels=sum(p["pixels"] for p in rawpairs),
                final_overlap_pixels=sum(p["pixels"] for p in finalpairs),
                max_shift_frame=max(p["shift_frame"] for p in rows),
                depth_order_reversals=sum(p["reversed"] for p in depthorder),
            )
        )
        (ROOT / "layout-summary.json").write_text(json.dumps(summaries, indent=2))
        print("LAYOUT", summaries[-1], flush=True)
