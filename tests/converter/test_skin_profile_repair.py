import numpy as np
import pytest

from converter.skin_profile_repair import VERSION, validate_plan, diffuse_arm_weights, correct_rest_surface


def tetra():
    return (np.array([[0.,0,0], [1.,0,0], [0.,1,0], [0.,0,1]]),
            np.array([[0,2,1], [0,1,3], [1,2,3], [2,0,3]]))


def plan_fixture():
    n, f = tetra()
    w = np.tile([.5, .5], (4, 1))
    snapshot = n.copy()
    snapshot[:, 1] += 3e-7
    corrected = snapshot.copy()
    corrected[1, 0] += .0004
    plan = dict(version=VERSION, source_sha256="a"*64, asset_sha256="b"*64,
        neutral=n.copy(), faces=f.copy(), repaired_rest=snapshot,
        corrected_rest=corrected, original_weights=w.copy(), weights=w.copy(),
        right_arm_clearance_degrees=6.)
    args = dict(neutral=n, repaired_rest=n.copy(), faces=f, original_weights=w,
                source_sha256="a"*64, asset_sha256="b"*64)
    return plan, args


def test_roundtrip_noise_never_becomes_an_inactive_vertex_edit():
    plan, args = plan_fixture()
    weights, rest, angle = validate_plan(plan, **args)
    assert np.array_equal(rest[[0,2,3]], args["neutral"][[0,2,3]])
    assert rest[1, 0] == pytest.approx(1.0004)
    assert np.array_equal(weights, args["original_weights"])
    assert angle == 6.


@pytest.mark.parametrize("change", ["source", "large_shift", "large_rotation", "large_weight", "nonfinite"])
def test_plan_rejects_changed_identity_or_authoring_bounds(change):
    plan, args = plan_fixture()
    if change == "source": plan["source_sha256"] = "c"*64
    if change == "large_shift": plan["corrected_rest"][1, 0] += .01
    if change == "large_rotation": plan["right_arm_clearance_degrees"] = 12.1
    if change == "large_weight": plan["weights"][0] = [1.1, -.1]
    if change == "nonfinite": plan["corrected_rest"][0, 0] = np.nan
    with pytest.raises(ValueError): validate_plan(plan, **args)


def test_diffusion_preserves_inactive_component_and_input():
    n, f = tetra()
    faces = np.concatenate([f, f+4])
    names = [side+bone for side in ("Left", "Right") for bone in ("Arm", "ForeArm", "Shoulder")]+["Hips"]
    w = np.zeros((8, 7))
    w[:4, 0] = [1, .8, .6, .4]
    w[:4, 1] = 1-w[:4, 0]
    w[4:, 6] = 1
    original = w.copy()
    out, report = diffuse_arm_weights(w, faces, names)
    assert np.array_equal(w, original)
    assert np.array_equal(out[4:], original[4:])
    assert np.allclose(out.sum(1), 1)
    assert np.all(out >= 0)
    assert report["changed_vertex_count"] == 4


def test_planner_cannot_certify_or_ignore_existing_crossings():
    n, f = tetra()
    n = np.concatenate([n, n+.1])
    f = np.concatenate([f, f+4])
    normals = np.cross(n[f[:,1]]-n[f[:,0]], n[f[:,2]]-n[f[:,0]])
    maps = np.repeat(np.eye(4)[None], len(n), axis=0)
    directors = np.repeat(normals[:,None,:], 3, axis=1)
    with pytest.raises(ValueError, match="no_safe_candidate"):
        correct_rest_surface(n, n, f, [("body", maps, directors)])


def test_successful_planning_still_requires_exported_fbx_verification():
    n, f = tetra()
    normals = np.cross(n[f[:,1]]-n[f[:,0]], n[f[:,2]]-n[f[:,0]])
    maps = np.repeat(np.eye(4)[None], len(n), axis=0)
    result, report = correct_rest_surface(n, n, f,
        [("body", maps, np.repeat(normals[:,None,:], 3, axis=1))])
    assert np.array_equal(result, n)
    assert report["exported_fbx_certified"] is False
    assert all(a["status"] == "mesh_validated" for a in report["planning_audits"].values())
