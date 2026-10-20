"""Explicit offline body experiment: original crop -> visibility -> observed contours.

Not registered with API/BFF. Callers provide detector coordinates and verified
ownership. Remote visibility and local segmentation are dependency-injected.
"""

from __future__ import annotations
import copy
import numpy as np
from .rough_evidence import PARTS, guide_image, parse_visibility, extract_contours
from .rough_segmentation import extract_mask_contours
from .rough_matching import compare_rough


class RoughBodyPipeline:
    def __init__(self, *, segmenter, visibility_client=None):
        self.segmenter = segmenter
        self.visibility_client = visibility_client

    def observe(self, image, case, *, visibility=None):
        case = copy.deepcopy(case)
        unknown = dict(
            parts={
                p: dict(state="uncertain", evidence="ownership unverified")
                for p in PARTS
            },
            ownership_ambiguous=True,
        )
        if not case["ownership_verified"]:
            visibility = unknown
            segments = {}
            whole = None
        else:
            if visibility is None:
                if self.visibility_client is None:
                    raise ValueError("visibility_provider_missing")
                guide = guide_image(image, case["points"], case["mask"])
                visibility = self.visibility_client.analyze(
                    [(case["case_id"], image, guide)]
                )[case["case_id"]]
            # Validate supplied/cached observations through the same strict parser.
            raw = dict(
                person_id=case["case_id"],
                ownership_ambiguous=visibility["ownership_ambiguous"],
                parts={str(i): visibility["parts"][p] for i, p in enumerate(PARTS, 1)},
            )
            visibility = parse_visibility({"people": [raw]}, [case["case_id"]])[
                case["case_id"]
            ]
            if visibility["ownership_ambiguous"]:
                segments = {}
                whole = None
            else:
                segments, whole = self.segmenter.segment(
                    image,
                    case["points"],
                    case["mask"],
                    visibility,
                    case.get("other_points", ()),
                )
        args = dict(
            ownership_verified=case["ownership_verified"],
            other_points=case.get("other_points", ()),
        )
        e, observed, excluded = extract_mask_contours(
            image, case["points"], case["mask"], visibility, segments, **args
        )
        # Strict unambiguous ink pairs can survive when the segmentation model fails.
        # This fallback never completes a hidden/uncertain part or uses a body prior.
        ink, ink_observed, _, _ = extract_contours(
            image, case["points"], case["mask"], visibility, **args
        )
        for part, row in e["parts"].items():
            if row["accepted"]:
                row["source"] = "segmentation_and_original_edges"
            elif ink["parts"].get(part, {}).get("accepted"):
                e["parts"][part] = {
                    **ink["parts"][part],
                    "source": "unambiguous_original_edge_pairs",
                }
        e["accepted_parts"] = [p for p, r in e["parts"].items() if r["accepted"]]
        observed = np.maximum(observed, ink_observed)
        return dict(**case, visibility=visibility, segmentation=e), dict(
            observed=observed, excluded_region=excluded, person=whole, segments=segments
        )

    def compare(self, case, records, assets, jobs):
        return compare_rough(case, records, assets, jobs)
