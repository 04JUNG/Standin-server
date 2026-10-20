"""Model text is evidence, never authority to execute files or select identities."""

import json, math
from .plan import pid


def validate_evidence(ctx, ids):
    for item in json.loads((ctx.work / "inputs.json").read_text()):
        folder = ctx.work / item["cut_id"]
        surface = json.loads((folder / "surfaces-review.json").read_text())["analysis"]
        rows = surface.get("people", [])
        if len(rows) != len(ids) or {pid(p["person_index"]) for p in rows} != set(ids):
            raise ValueError("fresh surface identities incomplete")
        headings = {
            "front",
            "front_left",
            "front_right",
            "left",
            "right",
            "back_left",
            "back_right",
            "back",
            "unknown",
        }
        for p in rows:
            b = p.get("head_bbox")
            if (
                not isinstance(b, list)
                or len(b) != 4
                or not all(
                    isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1000
                    for x in b
                )
                or b[0] >= b[2]
                or b[1] >= b[3]
            ):
                raise ValueError("invalid inferred head box")
            if (
                p.get("torso_surface") not in ("front", "side", "back")
                or p.get("torso_heading") not in headings
                or p.get("head_heading") not in headings
            ):
                raise ValueError("invalid inferred direction")
        roles = json.loads((folder / "roles-review.json").read_text())["analysis"]
        for rel in roles.get("relations", []):
            if pid(rel["actor"]) not in ids or pid(rel["target"]) not in ids:
                raise ValueError("unknown relation participant")
        for i in ids:
            a = json.loads((folder / f"person_{i}-analysis.json").read_text())[
                "analysis"
            ]
            if pid(a["person_index"]) != i:
                raise ValueError("actor identity changed")
        for hand in json.loads((folder / "localized-hands.json").read_text())["hands"]:
            a = hand["analysis"]
            if not a.get("visible"):
                continue
            values = [a.get("x"), a.get("y"), *(a.get("bbox") or [])]
            if len(values) != 6 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1000
                for v in values
            ):
                raise ValueError("invalid inferred hand coordinates")
            if values[2] >= values[4] or values[3] >= values[5]:
                raise ValueError("invalid inferred hand box")


def validate_layout(layout):
    """Never release a scene whose final separation/order gate failed."""
    if any(p["pixels"] for p in layout["final_overlap_pairs"]):
        raise ValueError("unresolved visible silhouette overlap")
    if any(p["reversed"] for p in layout["depth_order"]):
        raise ValueError("scene depth order reversed")
    if any(p["scale"] != 1.0 for p in layout["people"]):
        raise ValueError("physical body scale changed")
