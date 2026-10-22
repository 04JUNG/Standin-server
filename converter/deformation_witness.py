"""Immutable finite LBS witnesses for a discrete shell orientation test.

Inputs come from the original neutral mesh, weights and BEFORE-rest-rewrite pose
matrices. No delivered mesh normals are used as a reference. This is an exact
finite vertex-replay and conservative transported-director test, not a continuous
volume-Jacobian, anatomical, collision or arbitrary-pose certification. DQS and
unrepresented modifier/envelope/vertex-group attenuation are unsupported.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np

VERSION = "finite-normalized-lbs-shell-witness-v1"
AREA_RATIO_MIN = 0.001
MAX_VERTEX_TOLERANCE_M = 1e-5  # Existing FBX roundtrip contract: 0.01 mm.
MIN_SINGULAR_VALUE = 1e-8


class WitnessValidationError(ValueError):
    pass


def _fail(reason):
    raise WitnessValidationError(reason)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _array_sha(value, dtype="<f8"):
    array = np.ascontiguousarray(value, dtype=dtype)
    return _digest({"shape": list(array.shape), "dtype": dtype,
                    "data_sha256": hashlib.sha256(array.tobytes()).hexdigest()})


def _vertices(value, label):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array) or not np.isfinite(array).all():
        _fail(label + "_invalid")
    return array


def _faces(value, count):
    raw = np.asarray(value)
    if (raw.ndim != 2 or raw.shape[1] != 3 or not len(raw) or raw.dtype.kind not in "iu"
            or raw.min() < 0 or raw.max() >= count):
        _fail("topology_invalid")
    return np.asarray(raw, dtype=np.int64)


def _topology_sha(faces, count):
    # Matches converter.mesh_validation.topology_sha256 without importing it.
    return _digest({"vertices": int(count), "triangles": faces.tolist()})


def _normals(vertices, faces):
    tri = vertices[faces]
    return np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])


def _affine(value, label, batched=False):
    array = np.asarray(value, dtype=float)
    shape_ok = (array.ndim == 3 and array.shape[1:] == (4,4) and len(array)) if batched else array.shape == (4,4)
    if (not shape_ok or not np.isfinite(array).all()
            or not np.allclose(array[..., 3, :], [0.,0.,0.,1.], atol=1e-12, rtol=0)):
        _fail(label + "_invalid")
    return array


def _skin_maps(weights, transforms, vertex_count, deformation):
    if deformation != "lbs":
        _fail("deformation_unsupported")
    transforms = _affine(transforms, "bone_world_transforms", batched=True)
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (vertex_count, len(transforms)) or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        _fail("skin_weights_invalid")
    totals = weights.sum(axis=1)
    if np.any(totals <= 0) or np.any(np.abs(totals-1) > 1e-4):
        _fail("skin_weights_not_normalized")
    # Blender armature weights are normalized. Pin this rule in the witness
    # version; actual export still has to match the independently replayed vertices.
    normalized = weights / totals[:, None]
    maps = np.einsum("nb,bij->nij", normalized, transforms)
    linear = maps[:, :3, :3]
    determinant = np.linalg.det(linear)
    if not np.isfinite(determinant).all() or np.any(determinant <= 0):
        _fail("nonpositive_skin_determinant")
    singular_values = np.linalg.svd(linear, compute_uv=False)
    if not np.isfinite(singular_values).all() or np.any(singular_values[:, -1] <= MIN_SINGULAR_VALUE):
        _fail("near_singular_skin_transform")
    # det > 0 is checked separately: a cofactor alone would follow a reflected
    # transform and could hide an inverted element.
    cofactor = determinant[:,None,None] * np.linalg.inv(linear).swapaxes(1,2)
    return maps, cofactor, float(determinant.min())


def _apply(maps, vertices):
    return np.einsum("nij,nj->ni", maps[:, :3, :3], vertices) + maps[:, :3, 3]


def _shell_check(vertices, faces, directors, areas):
    normals = _normals(vertices, faces)
    double_areas = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or not np.isfinite(double_areas).all():
        _fail("shell_geometry_nonfinite")
    if (directors.shape != (len(faces), 3, 3) or areas.shape != (len(faces),)
            or not np.isfinite(directors).all() or not np.isfinite(areas).all()
            or np.any(np.linalg.norm(directors, axis=2) <= 0) or np.any(areas <= 0)):
        _fail("shell_reference_invalid")
    # All three vertex fields must agree. An average can hide one inverted corner.
    if np.any(np.einsum("fi,fvi->fv", normals, directors) <= 0):
        _fail("discrete_shell_orientation_reversed")
    if np.any(double_areas / areas < AREA_RATIO_MIN):
        _fail("discrete_shell_area_collapse")


@dataclass(frozen=True)
class DeformationWitness:
    source_asset_sha256: str
    profile_sha256: str
    source_sha256: str
    topology_sha256: str
    stage: str
    stage_input_sha256: str
    input_sha256: str
    previous_witness_sha256: str
    vertex_count: int
    face_count: int
    determinant_min: float
    vertex_tolerance_m: float
    _vertices_bytes: bytes = field(repr=False)
    _faces_bytes: bytes = field(repr=False)
    _directors_bytes: bytes = field(repr=False)
    _areas_bytes: bytes = field(repr=False)
    witness_sha256: str = ""
    version: str = VERSION

    def __post_init__(self):
        digests = (self.source_asset_sha256, self.profile_sha256, self.source_sha256,
                   self.topology_sha256, self.stage_input_sha256, self.input_sha256)
        if (not all(_sha(x) for x in digests) or not isinstance(self.stage, str) or not self.stage
                or (self.previous_witness_sha256 and not _sha(self.previous_witness_sha256))
                or self.version != VERSION or type(self.vertex_count) is not int or self.vertex_count <= 0
                or type(self.face_count) is not int or self.face_count <= 0
                or not np.isfinite(self.determinant_min) or self.determinant_min <= 0
                or not np.isfinite(self.vertex_tolerance_m)
                or not 0 < self.vertex_tolerance_m <= MAX_VERTEX_TOLERANCE_M):
            _fail("witness_identity_invalid")
        sizes = ((self._vertices_bytes, self.vertex_count*3*8), (self._faces_bytes, self.face_count*3*8),
                 (self._directors_bytes, self.face_count*9*8), (self._areas_bytes, self.face_count*8))
        if any(not isinstance(data, bytes) or len(data) != size for data, size in sizes):
            _fail("witness_payload_invalid")
        faces = _faces(self.faces, self.vertex_count)
        vertices = _vertices(self.expected_vertices, "witness_vertices")
        if _topology_sha(faces, self.vertex_count) != self.topology_sha256:
            _fail("witness_topology_mismatch")
        _shell_check(vertices, faces, self.expected_directors, self.reference_double_areas)
        calculated = _digest(self.manifest())
        if self.witness_sha256 and self.witness_sha256 != calculated:
            _fail("witness_content_mismatch")
        object.__setattr__(self, "witness_sha256", calculated)

    @property
    def expected_vertices(self):
        return np.frombuffer(self._vertices_bytes, dtype="<f8").reshape(self.vertex_count,3)

    @property
    def faces(self):
        return np.frombuffer(self._faces_bytes, dtype="<i8").reshape(self.face_count,3)

    @property
    def expected_directors(self):
        return np.frombuffer(self._directors_bytes, dtype="<f8").reshape(self.face_count,3,3)

    @property
    def reference_double_areas(self):
        return np.frombuffer(self._areas_bytes, dtype="<f8")

    @property
    def digest(self):
        return self.witness_sha256

    def manifest(self):
        return {"version": self.version, "source_asset_sha256": self.source_asset_sha256,
                "profile_sha256": self.profile_sha256, "source_sha256": self.source_sha256,
                "topology_sha256": self.topology_sha256, "stage": self.stage,
                "stage_input_sha256": self.stage_input_sha256, "input_sha256": self.input_sha256,
                "previous_witness_sha256": self.previous_witness_sha256,
                "vertex_count": self.vertex_count, "face_count": self.face_count,
                "determinant_min": self.determinant_min, "vertex_tolerance_m": self.vertex_tolerance_m,
                "area_ratio_min": AREA_RATIO_MIN,
                "expected_vertices_sha256": hashlib.sha256(self._vertices_bytes).hexdigest(),
                "faces_sha256": hashlib.sha256(self._faces_bytes).hexdigest(),
                "directors_sha256": hashlib.sha256(self._directors_bytes).hexdigest(),
                "reference_areas_sha256": hashlib.sha256(self._areas_bytes).hexdigest()}


def _make(vertices, faces, directors, areas, *, source_asset_sha256, profile_sha256,
          source_sha256, stage, stage_input_sha256, input_sha256, previous_witness_sha256,
          determinant_min, vertex_tolerance_m):
    return DeformationWitness(source_asset_sha256, profile_sha256, source_sha256,
        _topology_sha(faces, len(vertices)), stage, stage_input_sha256, input_sha256,
        previous_witness_sha256, len(vertices), len(faces), determinant_min, vertex_tolerance_m,
        np.asarray(vertices,dtype="<f8").tobytes(), np.asarray(faces,dtype="<i8").tobytes(),
        np.asarray(directors,dtype="<f8").tobytes(), np.asarray(areas,dtype="<f8").tobytes())


def build_lbs_witness(*, neutral_vertices, faces, vertex_weights, bone_world_transforms,
                      mesh_world_transform, source_asset_sha256, profile_sha256, source_sha256,
                      stage, stage_input_sha256, corrected_rest_vertices=None,
                      vertex_tolerance_m=MAX_VERTEX_TOLERANCE_M, deformation="lbs"):
    """Replay LBS with world→world bone deformation maps and local rest vertices.

    D_b = arm_world @ pose_b @ inverse(rest_b) @ inverse(arm_world).
    mesh_world_transform maps neutral/corrected local mesh coordinates to world.
    Corrective rest geometry may NOT redefine neutral normals/areas.
    """
    neutral = _vertices(neutral_vertices, "neutral_vertices")
    faces = _faces(faces, len(neutral))
    mesh_world = _affine(mesh_world_transform, "mesh_world_transform")
    if np.linalg.det(mesh_world[:3,:3]) <= 0:
        _fail("reflected_mesh_world_unsupported")
    corrected = neutral if corrected_rest_vertices is None else _vertices(corrected_rest_vertices, "corrected_rest_vertices")
    if corrected.shape != neutral.shape:
        _fail("corrected_rest_topology_mismatch")
    neutral_world = neutral @ mesh_world[:3,:3].T + mesh_world[:3,3]
    corrected_world = corrected @ mesh_world[:3,:3].T + mesh_world[:3,3]
    reference_normals = _normals(neutral_world, faces)
    areas = np.linalg.norm(reference_normals, axis=1)
    neutral_directors = np.repeat(reference_normals[:,None,:],3,axis=1)
    _shell_check(corrected_world, faces, neutral_directors, areas)
    maps, cofactor, determinant_min = _skin_maps(vertex_weights, bone_world_transforms, len(neutral), deformation)
    expected = _apply(maps, corrected_world)
    directors = np.einsum("fvij,fj->fvi", cofactor[faces], reference_normals)
    input_sha256 = _digest({"neutral":_array_sha(neutral), "corrected":_array_sha(corrected),
        "weights":_array_sha(vertex_weights), "bone_world_transforms":_array_sha(bone_world_transforms),
        "mesh_world_transform":_array_sha(mesh_world), "deformation":deformation})
    return _make(expected, faces, directors, areas, source_asset_sha256=source_asset_sha256,
        profile_sha256=profile_sha256, source_sha256=source_sha256, stage=stage,
        stage_input_sha256=stage_input_sha256, input_sha256=input_sha256,
        previous_witness_sha256="", determinant_min=determinant_min,
        vertex_tolerance_m=vertex_tolerance_m)


def compose_lbs_witness(previous, *, faces, vertex_weights, bone_world_transforms,
                        stage, stage_input_sha256, deformation="lbs"):
    """Append a hand/other affine-LBS stage without replacing the body proof.

    Uses previous expected vertices, directors and ORIGINAL neutral areas. The
    caller must bind stage_input_sha256 to the actual input artifact and verify
    the final output against this witness. Composition never blesses a failed
    body witness or an unrelated observed input surface.
    """
    if not isinstance(previous, DeformationWitness) or _digest(previous.manifest()) != previous.digest:
        _fail("previous_witness_invalid")
    faces = _faces(faces, previous.vertex_count)
    if _topology_sha(faces, previous.vertex_count) != previous.topology_sha256:
        _fail("compose_topology_mismatch")
    maps, cofactor, determinant_min = _skin_maps(vertex_weights, bone_world_transforms,
                                                previous.vertex_count, deformation)
    expected = _apply(maps, previous.expected_vertices)
    directors = np.einsum("fvij,fvj->fvi", cofactor[faces], previous.expected_directors)
    input_sha256 = _digest({"previous_witness_sha256":previous.digest,
        "weights":_array_sha(vertex_weights), "bone_world_transforms":_array_sha(bone_world_transforms),
        "deformation":deformation})
    return _make(expected, faces, directors, previous.reference_double_areas,
        source_asset_sha256=previous.source_asset_sha256, profile_sha256=previous.profile_sha256,
        source_sha256=previous.source_sha256, stage=stage, stage_input_sha256=stage_input_sha256,
        input_sha256=input_sha256, previous_witness_sha256=previous.digest,
        determinant_min=min(previous.determinant_min,determinant_min),
        vertex_tolerance_m=previous.vertex_tolerance_m)


def verify_witness(witness, *, vertices, faces, source_asset_sha256, profile_sha256,
                   source_sha256, stage_input_sha256, stage=None, expected_witness_sha256=None):
    """Match the actual evaluated/exported world vertices and exact context."""
    if not isinstance(witness, DeformationWitness) or _digest(witness.manifest()) != witness.digest:
        _fail("witness_content_mismatch")
    if ((source_asset_sha256,profile_sha256,source_sha256,stage_input_sha256) !=
            (witness.source_asset_sha256,witness.profile_sha256,witness.source_sha256,witness.stage_input_sha256)
            or (stage is not None and stage != witness.stage)
            or (expected_witness_sha256 is not None and expected_witness_sha256 != witness.digest)):
        _fail("witness_identity_mismatch")
    actual = _vertices(vertices,"actual_vertices")
    faces = _faces(faces,len(actual))
    if _topology_sha(faces,len(actual)) != witness.topology_sha256:
        _fail("witness_topology_mismatch")
    maximum = float(np.linalg.norm(actual-witness.expected_vertices,axis=1).max())
    if maximum > witness.vertex_tolerance_m:
        _fail("witness_vertex_replay_mismatch")
    _shell_check(actual,faces,witness.expected_directors,witness.reference_double_areas)
    return {"version":VERSION,"witness_sha256":witness.digest,"max_vertex_error_m":maximum,
            "vertex_tolerance_m":witness.vertex_tolerance_m,"area_ratio_min":AREA_RATIO_MIN,
            "discrete_shell_orientation_verified":True,"continuous_volume_certificate":False}
