from dataclasses import FrozenInstanceError, replace
import numpy as np
import pytest
from converter.deformation_witness import (AREA_RATIO_MIN, MAX_VERTEX_TOLERANCE_M,
    WitnessValidationError, build_lbs_witness, compose_lbs_witness, verify_witness)

SHA = "a"*64
V = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
F = np.array([[0,1,2]])


def build(**changes):
    kwargs = dict(neutral_vertices=V, faces=F, vertex_weights=np.ones((3,1)),
        bone_world_transforms=np.eye(4)[None], mesh_world_transform=np.eye(4),
        source_asset_sha256=SHA, profile_sha256=SHA, source_sha256=SHA,
        stage="body", stage_input_sha256=SHA)
    kwargs.update(changes)
    return build_lbs_witness(**kwargs)


def verify(witness, vertices=None, **changes):
    kwargs = dict(vertices=witness.expected_vertices if vertices is None else vertices,
        faces=F, source_asset_sha256=SHA, profile_sha256=SHA, source_sha256=SHA,
        stage_input_sha256=witness.stage_input_sha256)
    kwargs.update(changes)
    return verify_witness(witness, **kwargs)


def rigid(axis, angle, translation=(0.,0.,0.)):
    theta=np.radians(angle); c,s=np.cos(theta),np.sin(theta)
    matrix=np.eye(4)
    if axis == "x": matrix[:3,:3]=[[1,0,0],[0,c,-s],[0,s,c]]
    if axis == "y": matrix[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
    if axis == "z": matrix[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
    matrix[:3,3]=translation
    return matrix


def test_rest_reference_is_immutable_and_separate_from_corrected_shape():
    neutral=V.copy(); corrected=V*np.array([.5,.5,1.])
    witness=build(neutral_vertices=neutral, corrected_rest_vertices=corrected)
    np.testing.assert_allclose(witness.expected_vertices, corrected)
    np.testing.assert_allclose(witness.reference_double_areas, [1.])
    np.testing.assert_allclose(witness.expected_directors, [[[0,0,1]]*3])
    original=witness.digest
    neutral[:]=100;corrected[:]=-100
    assert witness.digest == original
    np.testing.assert_allclose(witness.expected_vertices, V*.5)
    for array in (witness.expected_vertices,witness.faces,witness.expected_directors,witness.reference_double_areas):
        with pytest.raises(ValueError): array.setflags(write=True)
    with pytest.raises(FrozenInstanceError): witness.stage="changed"
    data=witness.expected_vertices.copy();data[:,0]+=.001
    with pytest.raises(WitnessValidationError,match="content"):
        replace(witness,_vertices_bytes=data.astype('<f8').tobytes())


def test_world_transform_order_translation_and_weighted_affine_replay():
    mesh=rigid("z",90,(2.,3.,4.))
    bone=rigid("y",90,(5.,6.,7.))
    witness=build(mesh_world_transform=mesh,bone_world_transforms=bone[None])
    # Explicit known world positions before/after world→world bone deformation.
    expected=np.array([[9.,9.,5.],[9.,10.,5.],[9.,9.,6.]])
    np.testing.assert_allclose(witness.expected_vertices, expected,atol=1e-12)
    assert verify(witness)["discrete_shell_orientation_verified"]
    translations=np.stack([rigid("z",0,(1.,0.,0.)),rigid("z",0,(0.,2.,0.))])
    blended=build(vertex_weights=np.tile([.25,.75],(3,1)),bone_world_transforms=translations)
    np.testing.assert_allclose(blended.expected_vertices,V+[.25,1.5,0.])


def test_three_corner_directors_block_flip_hidden_by_positive_average():
    rotations=np.stack([rigid("x",180),np.eye(4),np.eye(4)])
    # All three positions remain exactly V; only corner 0's transported reference
    # is reversed. The average director is +Z, so averaging would falsely pass.
    with pytest.raises(WitnessValidationError,match="orientation"):
        build(vertex_weights=np.eye(3),bone_world_transforms=rotations)


@pytest.mark.parametrize("linear,reason",[
    (np.diag([-1.,1.,1.]),"nonpositive"),
    (np.diag([0.,1.,1.]),"nonpositive"),
    (np.diag([1e-10,1.,1.]),"near_singular"),
])
def test_reflection_and_singularity_cannot_follow_cofactor_into_pass(linear,reason):
    matrix=np.eye(4);matrix[:3,:3]=linear
    with pytest.raises(WitnessValidationError,match=reason):
        build(bone_world_transforms=matrix[None])


def test_corrective_cannot_redefine_rest_to_hide_flip_or_area_collapse():
    with pytest.raises(WitnessValidationError,match="orientation"):
        build(corrected_rest_vertices=V[[0,2,1]])
    with pytest.raises(WitnessValidationError,match="area_collapse"):
        build(corrected_rest_vertices=V*np.array([AREA_RATIO_MIN*.5,1.,1.]))
    collapsed=np.eye(4);collapsed[0,0]=AREA_RATIO_MIN*.5
    with pytest.raises(WitnessValidationError,match="area_collapse"):
        build(bone_world_transforms=collapsed[None])


def test_hand_stage_composes_body_reference_without_resetting_neutral_area():
    body_matrix=rigid("x",70,(1.,2.,3.))
    hand_matrix=rigid("z",80,(4.,5.,6.))
    body=build(bone_world_transforms=body_matrix[None])
    hands=compose_lbs_witness(body,faces=F,vertex_weights=np.ones((3,1)),
        bone_world_transforms=hand_matrix[None],stage="hands",stage_input_sha256="b"*64)
    composed=hand_matrix@body_matrix
    np.testing.assert_allclose(hands.expected_vertices,V@composed[:3,:3].T+composed[:3,3],atol=1e-12)
    np.testing.assert_allclose(hands.reference_double_areas,body.reference_double_areas)
    np.testing.assert_allclose(hands.expected_directors,
        np.einsum('ij,fvj->fvi',hand_matrix[:3,:3],body.expected_directors),atol=1e-12)
    assert hands.previous_witness_sha256 == body.digest
    assert hands.digest != body.digest
    assert verify(hands)["continuous_volume_certificate"] is False
    # Cumulative compression remains measured against ORIGINAL neutral area,
    # rather than rebasing the area in each new stage to bypass 0.001.
    scale=np.eye(4);scale[0,0]=.02
    first=build(bone_world_transforms=scale[None])
    with pytest.raises(WitnessValidationError,match="area_collapse"):
        compose_lbs_witness(first,faces=F,vertex_weights=np.ones((3,1)),
            bone_world_transforms=scale[None],stage="hands",stage_input_sha256="b"*64)


def test_identity_topology_and_expected_vertex_tolerance_are_all_pinned():
    witness=build()
    assert verify(witness,vertices=V+MAX_VERTEX_TOLERANCE_M*.5)["max_vertex_error_m"] <= MAX_VERTEX_TOLERANCE_M
    with pytest.raises(WitnessValidationError,match="replay"):
        verify(witness,vertices=V+[0.,0.,2*MAX_VERTEX_TOLERANCE_M])
    for field in ("source_asset_sha256","profile_sha256","source_sha256","stage_input_sha256"):
        with pytest.raises(WitnessValidationError,match="identity"):
            verify(witness,**{field:"b"*64})
    with pytest.raises(WitnessValidationError,match="identity"):
        verify(witness,expected_witness_sha256="b"*64)
    with pytest.raises(WitnessValidationError,match="topology"):
        verify(witness,faces=F[:,[0,2,1]])
    with pytest.raises(WitnessValidationError,match="topology"):
        compose_lbs_witness(witness,faces=F[:,[0,2,1]],vertex_weights=np.ones((3,1)),
            bone_world_transforms=np.eye(4)[None],stage="hands",stage_input_sha256="b"*64)
    assert build(stage_input_sha256="b"*64).digest != witness.digest
    with pytest.raises(WitnessValidationError,match="identity"):
        build(vertex_tolerance_m=MAX_VERTEX_TOLERANCE_M*2)


@pytest.mark.parametrize("changes",[
    {"deformation":"dqs"},
    {"vertex_weights":np.ones((3,1))*2},
    {"vertex_weights":np.array([[1.],[0.],[1.]])},
    {"vertex_weights":np.array([[1.],[-1.],[1.]])},
    {"bone_world_transforms":np.full((1,4,4),np.nan)},
    {"neutral_vertices":np.full((3,3),np.nan)},
    {"faces":np.array([[0.,1.,2.]])},
])
def test_unrepresented_or_nonfinite_inputs_fail_closed(changes):
    with pytest.raises(WitnessValidationError): build(**changes)



def test_weight_gradient_translation_flip_is_blocked_despite_positive_linear_determinants():
    maps=np.repeat(np.eye(4)[None],3,axis=0)
    maps[1,:3,3]=[-1.,1.,0.];maps[2,:3,3]=[1.,-1.,0.]
    with pytest.raises(WitnessValidationError,match="orientation"):
        build(vertex_weights=np.eye(3),bone_world_transforms=maps)


@pytest.mark.parametrize("mutate,reason", [
    (lambda vertices:vertices[[0,2,1]],"orientation"),
    (lambda vertices:vertices*np.array([1.,AREA_RATIO_MIN*.5,1.]),"area_collapse"),
])
def test_export_tolerance_does_not_waive_fold_or_collapse_of_small_triangles(mutate,reason):
    small=V*1e-6
    witness=build(neutral_vertices=small)
    actual=mutate(small)
    assert np.linalg.norm(actual-small,axis=1).max() < MAX_VERTEX_TOLERANCE_M
    with pytest.raises(WitnessValidationError,match=reason):
        verify(witness,vertices=actual)
