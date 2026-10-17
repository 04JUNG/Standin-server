"""Paired evaluation of frozen person inputs; stability is not accuracy."""


def compare_runs(before, after, *, labels=None, focus_cases=()):
    labels = labels or {}
    for key in ("protocol_sha256", "catalog_sha256"):
        if not before.get(key) or before[key] != after.get(key):
            raise ValueError("comparison_" + key + "_mismatch")

    def index(run):
        rows = run["rows"]
        result = {r["case_id"]: r for r in rows}
        if not result or len(result) != len(rows):
            raise ValueError("comparison_duplicate_or_empty_cases")
        return result

    old, new = index(before), index(after)
    if old.keys() != new.keys():
        raise ValueError("comparison_case_mismatch")
    focus = set(focus_cases)
    if not focus <= old.keys() or not labels.keys() <= old.keys():
        raise ValueError("comparison_unknown_case")
    if any(v not in ("male", "female") for v in labels.values()):
        raise ValueError("comparison_invalid_label")
    rows = []
    fields = ("observed_family", "recommended_family", "recommended_body_id")
    for ident, current in new.items():
        previous = old[ident]
        if not current.get("crop_sha256") or current["crop_sha256"] != previous.get(
            "crop_sha256"
        ):
            raise ValueError("comparison_crop_mismatch")
        for key in (
            "image_sha256",
            "context_sha256",
            "target_box",
            "crop_box",
            "input_version",
        ):
            if current.get("input_metadata", {}).get(key) != previous.get(
                "input_metadata", {}
            ).get(key):
                raise ValueError("comparison_input_mismatch")
        a, b = previous["recommendation"], current["recommendation"]
        expected = labels.get(ident)
        rows.append(
            dict(
                case_id=ident,
                focus=ident in focus,
                before={k: a[k] for k in fields},
                after={k: b[k] for k in fields},
                changed={k: a[k] != b[k] for k in fields},
                provider_error_before=previous.get("provider_error"),
                provider_error_after=current.get("provider_error"),
                expected_family=expected,
                before_correct=(
                    a["recommended_family"] == expected if expected else None
                ),
                after_correct=b["recommended_family"] == expected if expected else None,
            )
        )

    def summarize(selected):
        errors = sum(
            bool(r["provider_error_before"] or r["provider_error_after"])
            for r in selected
        )
        return dict(
            total=len(selected),
            changed={k: sum(r["changed"][k] for r in selected) for k in fields},
            provider_error_pairs=errors,
            confirmed_total=sum(r["expected_family"] is not None for r in selected),
            confirmed_improvements=sum(
                r["before_correct"] is False and r["after_correct"] is True
                for r in selected
            ),
            confirmed_regressions=sum(
                r["before_correct"] is True and r["after_correct"] is False
                for r in selected
            ),
            unlabelled_recommendation_changes=sum(
                r["expected_family"] is None and r["changed"]["recommended_body_id"]
                for r in selected
            ),
        )

    return dict(
        metric="paired_stability_not_accuracy",
        note="Only user-confirmed family labels measure correctness. Unlabelled or physique changes require review; equal aggregate counts do not establish non-regression.",
        all=summarize(rows),
        other_cases=summarize([r for r in rows if not r["focus"]]),
        focus_cases=sorted(focus),
        rows=rows,
    )
