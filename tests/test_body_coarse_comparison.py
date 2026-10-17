import copy
import pytest
from scripts.body_coarse_comparison import compare_runs


def run():
    return dict(
        protocol_sha256="protocol",
        catalog_sha256="catalog",
        rows=[
            dict(
                case_id=ident,
                crop_sha256=ident + "-crop",
                input_metadata=dict(
                    image_sha256=ident,
                    context_sha256=ident + "-context",
                    target_box=[1, 2, 3, 4],
                    crop_box=[0, 0, 5, 5],
                    input_version="v1",
                ),
                provider_error=None,
                recommendation=dict(
                    observed_family=family,
                    recommended_family=family,
                    recommended_body_id=family + "-base",
                ),
            )
            for ident, family in [("a", "male"), ("b", "female"), ("c", "male")]
        ],
    )


def test_paired_comparison_detects_swaps_hidden_by_equal_counts():
    before, after = run(), run()
    after["rows"][0]["recommendation"], after["rows"][1]["recommendation"] = (
        after["rows"][1]["recommendation"],
        after["rows"][0]["recommendation"],
    )
    snapshot = copy.deepcopy(after)
    result = compare_runs(
        before, after, labels={"a": "female", "b": "female"}, focus_cases=["a"]
    )
    assert result["all"]["changed"]["recommended_family"] == 2
    assert result["all"]["confirmed_improvements"] == 1
    assert result["other_cases"]["confirmed_regressions"] == 1
    assert result["other_cases"]["total"] == 2
    assert after == snapshot


def test_unlabelled_preset_change_and_provider_failure_are_not_success():
    before, after = run(), run()
    after["rows"][2]["recommendation"]["recommended_body_id"] = "male-muscular"
    after["rows"][2]["provider_error"] = "TimeoutError"
    result = compare_runs(before, after)
    assert result["all"]["changed"]["recommended_family"] == 0
    assert result["all"]["unlabelled_recommendation_changes"] == 1
    assert result["all"]["provider_error_pairs"] == 1
    assert result["all"]["confirmed_total"] == 0
    assert result["metric"] == "paired_stability_not_accuracy"


@pytest.mark.parametrize("field", ["protocol_sha256", "catalog_sha256"])
def test_comparison_rejects_different_protocol_or_catalog(field):
    a, b = run(), run()
    b[field] = "different"
    with pytest.raises(ValueError):
        compare_runs(a, b)


@pytest.mark.parametrize(
    "field",
    ["image_sha256", "context_sha256", "target_box", "crop_box", "input_version"],
)
def test_comparison_rejects_different_frozen_input(field):
    a, b = run(), run()
    b["rows"][0]["input_metadata"][field] = "different"
    with pytest.raises(ValueError, match="comparison_input_mismatch"):
        compare_runs(a, b)


@pytest.mark.parametrize("change", ["duplicate", "missing", "crop", "focus", "label"])
def test_comparison_rejects_incomplete_identity_or_unknown_scope(change):
    a, b = run(), run()
    options = {}
    if change == "duplicate":
        b["rows"].append(copy.deepcopy(b["rows"][0]))
    elif change == "missing":
        b["rows"].pop()
    elif change == "crop":
        b["rows"][0]["crop_sha256"] = "different"
    elif change == "focus":
        options["focus_cases"] = ["missing"]
    else:
        options["labels"] = {"a": "unknown"}
    with pytest.raises(ValueError):
        compare_runs(a, b, **options)
