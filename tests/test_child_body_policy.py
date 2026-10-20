"""Child is an explicit model choice, never an age guessed from rough style."""

import copy
import pytest

from src.experimental.body_matching.child_policy import is_child_asset, REASON
from src.experimental.body_matching.selection import compare_body_shapes
from src.experimental.body_matching.schema import parse_person
from tests.test_body_matching import attributes, person_payload


def assets():
    base = dict(
        body_id="adult-base",
        body_version="v1",
        asset_sha256="a" * 64,
        rig_version="r1",
        measurement_version="m1",
        attributes=attributes(),
        metadata={"age_design": "young_adult"},
        projections=[],
    )
    child = copy.deepcopy(base)
    child.update(body_id="body-child", metadata={"age_design": "child"})
    child["attributes"].update(
        head_ratio_class="h3", body_build="small_frame", limb_proportion="short"
    )
    return [base, child]


@pytest.mark.parametrize(
    "condition",
    ["adult", "stylized", "crouched", "occluded", "ambiguous", "clear_child_design"],
)
def test_child_never_enters_image_recommendations_even_with_confident_vlm_values(
    condition,
):
    raw = person_payload("p")
    raw["attributes"]["head_ratio_class"].update(
        value="h3", evidence="large illustrated head"
    )
    raw["attributes"]["body_build"].update(
        value="small_frame", evidence="small drawn figure"
    )
    raw["attributes"]["limb_proportion"].update(
        value="short", evidence="short visible legs"
    )
    if condition == "crouched":
        raw["foreshortening"] = True
    if condition == "occluded":
        raw["clothing_occlusion"] = True
        raw["coverage"]["torso"] = "unknown"
    if condition == "ambiguous":
        raw["ownership_ambiguous"] = True
    if condition == "stylized":
        raw["surface_detail"] = "omitted"
    observation = {**parse_person(raw, "p"), "geometry": {}}
    catalog = assets()
    before = copy.deepcopy(catalog)
    decision = compare_body_shapes(catalog, observation, default_body_id="body-child")
    assert decision["suggested_body_id"] == "adult-base"
    assert decision["auto_body_id"] != "body-child"
    assert all(r["body_id"] != "body-child" for r in decision["candidates"])
    assert REASON in decision["reason_codes"]
    assert catalog == before  # catalog and manual availability are never retired


def test_child_only_catalog_requires_selection_instead_of_reintroducing_default():
    decision = compare_body_shapes(
        assets()[1:],
        {**parse_person(person_payload("p"), "p"), "geometry": {}},
        default_body_id="body-child",
    )
    assert decision["suggested_body_id"] is None
    assert decision["auto_body_id"] is None
    assert decision["candidates"] == [] and decision["diagnostic"] == REASON


def test_child_asset_remains_catalog_eligible_for_explicit_selection():
    from src.experimental.body_matching.catalog import Catalog

    child = {
        **assets()[1],
        "availability": "eligible",
        "supported_pose_ids": ["pose-a"],
    }
    catalog = Catalog("fixture", "a" * 64, child["body_id"], (child,), ())
    assert catalog.eligible(["pose-a"]) == [child]


@pytest.mark.parametrize(
    "asset",
    [
        {"body_id": "body-child"},
        {"body_id": "child"},
        {"character_id": "rig-kid-v2"},
        {"body_id": "opaque-asset", "metadata": {"age_design": "child"}},
    ],
)
def test_catalog_child_identity_not_image_id_or_style(asset):
    assert is_child_asset(asset)


def test_compact_adult_and_ordinary_adult_results_remain_unchanged():
    catalog = assets()[:1]
    compact = copy.deepcopy(catalog[0])
    compact.update(body_id="adult-toon64", metadata={"age_design": None})
    compact["attributes"]["head_ratio_class"] = "h6"
    catalog.append(compact)
    assert not is_child_asset(compact)
    observation = {**parse_person(person_payload("p"), "p"), "geometry": {}}
    prior = compare_body_shapes(catalog, observation, default_body_id="adult-base")
    guarded = compare_body_shapes(
        catalog + assets()[1:], observation, default_body_id="adult-base"
    )
    assert guarded["suggested_body_id"] == prior["suggested_body_id"]
    assert guarded["auto_body_id"] == prior["auto_body_id"]
    assert guarded["candidates"] == prior["candidates"]
