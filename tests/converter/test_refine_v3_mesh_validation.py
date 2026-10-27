from dataclasses import replace
import hashlib
import numpy as np
import pytest
from converter.mesh_validation import MeshPolicy, audit_mesh, topology_sha256, triangle_intersection, triangle_relations
from converter.artifact_receipt import issue_receipt, verify_receipt
from converter.final_artifact import VERSION as INTEGRITY_VERSION

SHA = "a"*64
V = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
F = np.array([[0,2,1], [0,1,3], [0,3,2], [1,2,3]])


def audit(vertices=V, faces=F, **kwargs):
    tri = np.asarray(vertices)[faces]
    normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    defaults = dict(policy=MeshPolicy(SHA, topology_sha256(faces, len(vertices)), "fixture-v1",
                                     tuple("skin" for _ in faces)),
                    profile_sha256=SHA, expected_normals=normals,
                    reference_double_areas=np.linalg.norm(normals, axis=1))
    defaults.update(kwargs)
    return audit_mesh(vertices, faces, **defaults)


def test_closed_separated_skin_is_validated_in_explicit_synthetic_policy():
    result = audit()
    assert result["status"] == "mesh_validated", result
    assert result["complete_collision_certificate"]
    separated = np.vstack((V, V+3))
    assert audit(separated, np.vstack((F,F+4)))["status"] == "mesh_validated"


def test_enclosed_component_without_crossing_is_blocked():
    vertices = np.vstack((V, V*.15+.15))
    result = audit(vertices, np.vstack((F,F+4)))
    assert result["status"] == "validation_blocked"
    assert "skin_component_containment" in result["hard_violations"]
    assert not result["triangle_intersections"]


def test_contact_annotation_cannot_exempt_intersecting_skin():
    vertices = np.vstack((V, V+.2))
    result = audit(vertices, np.vstack((F,F+4)))
    assert "skin_triangle_intersection" in result["hard_violations"]
    assert result["status"] == "validation_blocked"


def test_coplanar_overlap_and_adjacent_fold_cannot_hide_in_seam_exclusion():
    assert triangle_intersection([[0,0,0],[1,0,0],[0,1,0]], [[.1,.1,0],[.9,.1,0],[.1,.9,0]])
    vertices = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [.2,.2,0.], [.8,.2,0.]])
    result = audit(vertices, np.array([[0,1,2],[0,3,4]]))
    assert "skin_triangle_intersection" in result["hard_violations"]


def test_open_mesh_unsupported_fold_reference_and_full_body_are_unverified():
    for kwargs in ({"faces":F[:-1]}, {"policy":None}, {"profile_sha256":"b"*64},
                   {"expected_normals":None}, {"full_body":False}, {"seconds":0},
                   {"max_triangles":1}, {"max_pairs":0}):
        result = audit(**kwargs)
        assert result["status"] == "mesh_unverified", result
        assert not result["complete_collision_certificate"]


def test_folding_nonfinite_and_collapse_are_hard_failures():
    normals = -np.cross(V[F][:,1]-V[F][:,0],V[F][:,2]-V[F][:,0])
    assert "skin_folding" in audit(expected_normals=normals)["hard_violations"]
    vertices = V.copy(); vertices[0,0]=float("nan")
    assert audit(vertices)["status"] == "validation_blocked"
    vertices = V.copy(); vertices[1]=vertices[0]
    assert audit(vertices)["status"] == "validation_blocked"


def test_receipt_binds_final_bytes_and_revision_and_does_not_wait_for_preview():
    artifact = b"immutable FBX bytes"
    digest = hashlib.sha256(artifact).hexdigest()
    mesh = {**audit(), "full_body_artifact_sha256":digest}
    integrity = dict(version=INTEGRITY_VERSION, artifact_sha256=digest, scope="full",
                     skin_verified=True, bones=52, meshes=1, vertices=4, polygons=4)
    kwargs = dict(source_sha256=SHA, context_sha256=SHA, profile_sha256=SHA,
                  revision="r1", scope="full", mesh_audit=mesh, integrity_audit=integrity)
    receipt = issue_receipt(artifact, **kwargs)
    assert verify_receipt(receipt, artifact, revision="r1", context_sha256=SHA, profile_sha256=SHA)
    assert not verify_receipt(receipt, artifact+b"changed", revision="r1", context_sha256=SHA, profile_sha256=SHA)
    assert not verify_receipt(receipt, artifact, revision="r2", context_sha256=SHA, profile_sha256=SHA)
    for changes in ({"status":"mesh_unverified"}, {"full_body":False}, {"hard_violations":["intersection"]},
                    {"profile_sha256":"b"*64}):
        with pytest.raises(ValueError, match="mesh_unverified"):
            issue_receipt(artifact, **{**kwargs, "mesh_audit":{**mesh, **changes}})
    with pytest.raises(ValueError, match="reimport"):
        issue_receipt(artifact, **{**kwargs, "integrity_audit":{**integrity, "artifact_sha256":SHA}})


def test_safe_mesh_receipt_cannot_authorize_unrelated_or_cropped_final_bytes():
    artifact = b"checked full-body bytes"
    digest = hashlib.sha256(artifact).hexdigest()
    mesh = {**audit(), "full_body_artifact_sha256": digest}
    def integrity(data, scope):
        return dict(version=INTEGRITY_VERSION, artifact_sha256=hashlib.sha256(data).hexdigest(),
                    scope=scope, skin_verified=True, bones=52, meshes=1, vertices=4, polygons=4)
    for data, scope in ((artifact+b"replacement", "full"), (artifact, "head")):
        with pytest.raises(ValueError, match="derivation"):
            issue_receipt(data, source_sha256=SHA, context_sha256=SHA, profile_sha256=SHA,
                          revision="r1", scope=scope, mesh_audit=mesh,
                          integrity_audit=integrity(data, scope))


def test_batched_sat_matches_scalar_for_unrelated_triangles():
    random = np.random.default_rng(714)
    first = random.normal(size=(300,3,3))
    second = random.normal(size=(300,3,3))
    ids = np.arange(900).reshape(300,3)
    intersects, boundary, _ = triangle_relations(first, second, ids, ids+900)
    assert not boundary.any()
    assert intersects.tolist() == [triangle_intersection(a,b) for a,b in zip(first,second)]


def test_sharp_shared_edge_has_only_its_exact_topological_boundary():
    first = np.array([[[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]]])
    # Distinct planes intersect exactly on the common edge, even at a sharp crease.
    second = np.array([[[1.,0.,0.], [0.,0.,0.], [0.,1.,.01]]])
    intersects, boundary, _ = triangle_relations(first,second,[[0,1,2]],[[1,0,3]])
    assert boundary[0] and not intersects[0]


def test_shared_vertex_does_not_exempt_a_real_crossing_or_duplicate_face():
    first = np.array([[[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]]])
    second = np.array([[[0.,0.,0.], [.25,.25,-1.], [.25,.25,1.]]])
    intersects,boundary,_ = triangle_relations(first,second,[[0,1,2]],[[0,3,4]])
    assert intersects[0] and not boundary[0]
    intersects,boundary,_ = triangle_relations(first,first,[[0,1,2]],[[0,1,2]])
    assert intersects[0] and not boundary[0]


def test_all_three_independent_vertex_directors_must_agree():
    normals = np.cross(V[F][:,1]-V[F][:,0], V[F][:,2]-V[F][:,0])
    directors = np.repeat(normals[:,None,:],3,axis=1)
    directors[0,1] *= -1
    assert "skin_folding" in audit(expected_normals=directors)["hard_violations"]


def test_reference_direction_scaling_does_not_change_fold_or_area_threshold():
    normals = np.cross(V[F][:,1]-V[F][:,0], V[F][:,2]-V[F][:,0])
    for factor in (1e-9, 1., 1e9):
        assert audit(expected_normals=normals*factor)["status"] == "mesh_validated"
        assert "skin_folding" in audit(expected_normals=-normals*factor)["hard_violations"]


@pytest.mark.parametrize("width", [2e-7, 5e-7, 1e-6])
def test_thin_coplanar_overlap_cannot_be_erased_by_shared_boundary_inset(width):
    first = np.array([[[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]]])
    second = np.array([[[0.,0.,0.], [1.,width,0.], [1.,-1.,0.]]])
    crossing,boundary,_ = triangle_relations(first,second,[[0,1,2]],[[0,3,4]])
    assert crossing[0] and not boundary[0]


@pytest.mark.parametrize("shift", [0.,1e8])
def test_shared_edge_proof_is_translation_invariant_for_thin_overlap(shift):
    first = np.array([[[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]]])+[0,shift,0]
    second = np.array([[[1.,0.,0.], [0.,0.,0.], [.5,1e-6,0.]]])+[0,shift,0]
    crossing,boundary,_ = triangle_relations(first,second,[[0,1,2]],[[1,0,3]])
    assert crossing[0] and not boundary[0]


def test_world_coordinates_without_clearance_precision_abstain():
    assert "coordinate_precision_unsupported" in audit(vertices=V+1e10)["unverified_reasons"]
