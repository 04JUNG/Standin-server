from dataclasses import FrozenInstanceError, replace
import hashlib
import pytest
from src.target_profile import (TargetProfile, EquivalenceReceipt, TargetEvaluation,
    CertifiedTargetRegistry, TARGET_REGISTRY, certified_adapter)

SHA = "a"*64
SOURCE = hashlib.sha256(b"pose").hexdigest()


def profile():
    return TargetProfile("fixture", *([SHA]*9), revision="r1", hand_revision="h1", adapter_version="fixture-v1",
                         converter_version="fixture-converter", blender_build_hash="fixture-build")


class Adapter:
    coordinate_frame = "right-handed-z-up"
    def __init__(self):
        self.profile = profile()
        self.receipt = EquivalenceReceipt(self.profile.digest, "fixture-v1", (SOURCE,),
            "fixture-converter", "fixture-build", 1e-6, 0.0, 1e-5, SHA,
            evaluation_sha256s=((SOURCE, self.evaluate("pose").digest),))
    def evaluate(self, text):
        return TargetEvaluation(self.profile.digest, hashlib.sha256(text.encode()).hexdigest(),
                                tuple((0., 0., 0.) for _ in range(17)))


def test_exact_identity_all_shapes_hands_revisions_and_implementation():
    original = profile()
    for name in original.__dataclass_fields__:
        if name in ("units", "coordinates", "output_mode"):
            continue
        value = "b"*64 if name.endswith("sha256") else "changed"
        assert replace(original, **{name: value}).digest != original.digest
    with pytest.raises(FrozenInstanceError):
        original.revision = "r2"


def test_registry_has_no_production_support_and_finite_pose_envelope():
    assert TARGET_REGISTRY.resolve(SHA, SOURCE) is None
    registry, adapter = CertifiedTargetRegistry(), Adapter()
    registry.register(adapter)
    assert registry.resolve(adapter.profile.digest, SOURCE) is adapter
    assert registry.resolve(adapter.profile.digest, SHA) is None
    assert registry.evaluate(adapter.profile.digest, "pose").source_sha256 == SOURCE
    with pytest.raises(ValueError, match="unsupported"):
        registry.evaluate(adapter.profile.digest, "another candidate")
    assert not certified_adapter(adapter, adapter.profile.digest, SOURCE, require_patches=True)


def test_bad_certificates_mismatch_and_nonfinite_error_failclosed():
    adapter = Adapter()
    for changes in ({"max_joint_error_m": float("nan")}, {"max_joint_error_m": .1},
                    {"profile_sha256": SHA}, {"adapter_version": "wrong"},
                    {"supported_source_sha256s": ()}):
        adapter.receipt = replace(Adapter().receipt, **changes)
        assert not certified_adapter(adapter, adapter.profile.digest, SOURCE)
        with pytest.raises(ValueError):
            CertifiedTargetRegistry().register(adapter)


def test_reject_evaluation_hash_mismatch_and_uncertified_surface_patch():
    adapter, registry = Adapter(), CertifiedTargetRegistry()
    registry.register(adapter)
    adapter.evaluate = lambda text: TargetEvaluation(SHA, SOURCE, tuple((0.,0.,0.) for _ in range(17)))
    with pytest.raises(ValueError, match="identity"):
        registry.evaluate(adapter.profile.digest, "pose")
    with pytest.raises(ValueError):
        TargetEvaluation(SHA, SOURCE, tuple((float("nan"),0.,0.) for _ in range(17)))


def test_snapshot_adapter_exact_cache_content_and_unknown_rotations_abstain():
    from src.target_fk import SnapshotTargetAdapter
    adapter = Adapter()
    evaluation = adapter.evaluate("pose")
    exact = SnapshotTargetAdapter(adapter.profile, adapter.receipt, (evaluation,))
    assert exact.evaluate("pose") == evaluation
    with pytest.raises(ValueError, match="envelope"):
        exact.evaluate("changed rotations")
    wrong = replace(evaluation, coco17=tuple((1.,1.,1.) for _ in range(17)))
    with pytest.raises(ValueError, match="unverified"):
        SnapshotTargetAdapter(adapter.profile, adapter.receipt, (wrong,))
    with pytest.raises(ValueError, match="immutable"):
        replace(evaluation, patches=[])
