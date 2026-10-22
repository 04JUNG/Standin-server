import copy
import numpy as np
import pytest
from converter.neutral_mesh_repair import repair_neutral, MAX_SHIFT_M

V = np.array([[0.,0.,0.],[.001,0.,0.],[0.,.001,0.],[0.,0.,.001],
              [2.,2.,2.]])
F = np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
BASE = dict(full_body=True,hard_violations=["skin_triangle_intersection"],
            unverified_reasons=[],triangle_intersections=[[0,1]])
SAFE = dict(full_body=True,status="mesh_validated",complete_collision_certificate=True,
            hard_violations=[],unverified_reasons=[])


def test_repair_is_explicit_bounded_and_does_not_mutate_source_or_inactive_vertices():
    original = V.copy()
    result, report = repair_neutral(V,F,BASE,lambda _:copy.deepcopy(SAFE))
    assert np.array_equal(V,original)
    assert np.array_equal(result[4],original[4])
    assert report["max_shift_m"] <= MAX_SHIFT_M
    assert report["changed_vertices"] == [0,1,2,3]
    assert report["original_geometry_sha256"] != report["repaired_geometry_sha256"]


@pytest.mark.parametrize("changes", [{"unverified_reasons":["validation_deadline"]},
    {"hard_violations":["skin_folding"]},{"full_body":False}])
def test_incomplete_or_nonintersection_diagnosis_cannot_start_repair(changes):
    with pytest.raises(ValueError,match="complete_intersection"):
        repair_neutral(V,F,{**BASE,**changes},lambda _:pytest.fail("must not validate"))


@pytest.mark.parametrize("changes", [{"complete_collision_certificate":False},
    {"hard_violations":["skin_triangle_intersection"]},
    {"unverified_reasons":["validation_deadline"]},{"status":"mesh_unverified"}])
def test_partial_or_unverified_candidate_never_becomes_a_repaired_asset(changes):
    original = V.copy()
    with pytest.raises(ValueError,match="no_fully_validated"):
        repair_neutral(V,F,BASE,lambda _:{**SAFE,**changes})
    assert np.array_equal(V,original)


def test_large_deformations_are_rejected_without_relaxing_trust_budget():
    with pytest.raises(ValueError,match="no_fully_validated"):
        repair_neutral(V*100,F,BASE,lambda _:pytest.fail("exceeds trust budget"))
