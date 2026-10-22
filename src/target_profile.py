"""Fail-closed exact target identity and finite Converter equivalence contracts.

The production registry is intentionally empty. A changed shape value, hand,
asset, source pose or retarget implementation needs another certificate. A
certificate for sampled poses does not certify an entire rotation envelope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Protocol


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class TargetProfile:
    profile_id: str
    character_sha256: str
    shape_sha256: str
    rig_sha256: str
    rest_sha256: str
    bone_map_sha256: str
    hand_sha256: str
    retarget_sha256: str
    converter_manifest_sha256: str
    controller_sha256: str
    revision: str
    hand_revision: str
    adapter_version: str
    converter_version: str
    blender_build_hash: str
    # Explicit coordinate contract, never inferred from source bone lengths.
    units: str = "meters"
    coordinates: str = "right-handed-z-up"
    output_mode: str = "rigged_rest"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name.endswith("_sha256") and not _sha(value):
                raise ValueError(f"invalid {name}")
            if not isinstance(value, str) or not value:
                raise ValueError(f"missing {name}")
        if (self.units, self.coordinates, self.output_mode) != (
                "meters", "right-handed-z-up", "rigged_rest"):
            raise ValueError("unsupported target space")

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class EquivalenceReceipt:
    profile_sha256: str
    adapter_version: str
    supported_source_sha256s: tuple[str, ...]
    converter_version: str
    blender_build_hash: str
    max_joint_error_m: float
    max_patch_error_m: float
    tolerance_m: float
    # Hash of persisted independent Converter/reimport comparison evidence.
    evidence_sha256: str
    patch_verified: bool = False
    evaluation_sha256s: tuple[tuple[str, str], ...] = ()

    @property
    def verified(self) -> bool:
        errors = (self.max_joint_error_m, self.max_patch_error_m, self.tolerance_m)
        return (all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in errors)
                and self.tolerance_m > 0
                and max(self.max_joint_error_m, self.max_patch_error_m) <= self.tolerance_m
                and _sha(self.profile_sha256) and _sha(self.evidence_sha256)
                and isinstance(self.supported_source_sha256s, tuple)
                and bool(self.supported_source_sha256s)
                and all(_sha(x) for x in self.supported_source_sha256s)
                and bool(self.adapter_version) and bool(self.converter_version)
                and bool(self.blender_build_hash)
                and isinstance(self.evaluation_sha256s, tuple)
                and len(self.evaluation_sha256s) == len(self.supported_source_sha256s)
                and all(isinstance(pair, tuple) and len(pair) == 2 and all(_sha(x) for x in pair)
                        for pair in self.evaluation_sha256s)
                and {pair[0] for pair in self.evaluation_sha256s} == set(self.supported_source_sha256s)
                and len(set(self.supported_source_sha256s)) == len(self.supported_source_sha256s))


@dataclass(frozen=True)
class TargetEvaluation:
    """COCO17 and optional surface patches in world meters, Blender Z-up."""
    profile_sha256: str
    source_sha256: str
    coco17: tuple[tuple[float, float, float], ...]
    # (anchor name, position, normal); no wrist-as-palm substitution.
    patches: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...] = ()

    def __post_init__(self):
        if not _sha(self.profile_sha256) or not _sha(self.source_sha256):
            raise ValueError("invalid evaluation identity")
        if (not isinstance(self.coco17, tuple) or len(self.coco17) != 17
                or any(not isinstance(p, tuple) or len(p) != 3
                       or not all(math.isfinite(x) for x in p) for p in self.coco17)):
            raise ValueError("target requires finite immutable COCO17")
        if not isinstance(self.patches, tuple):
            raise ValueError("target patches must be immutable")
        for patch in self.patches:
            if not isinstance(patch, tuple) or len(patch) != 3:
                raise ValueError("invalid target patch")
            name, position, normal = patch
            if not isinstance(position, tuple) or not isinstance(normal, tuple):
                raise ValueError("target patches must be immutable")
            if (not name or len(position) != 3 or len(normal) != 3
                    or not all(math.isfinite(x) for x in (*position, *normal))
                    or abs(sum(x*x for x in normal) - 1) > 1e-5):
                raise ValueError("invalid target surface patch")


    @property
    def digest(self) -> str:
        return _digest(asdict(self))


class TargetAdapter(Protocol):
    profile: TargetProfile
    receipt: EquivalenceReceipt

    def evaluate(self, source_text: str) -> TargetEvaluation: ...


def certified_adapter(adapter: TargetAdapter | None, profile_sha256: str,
                      source_sha256: str, *, require_patches: bool = False) -> bool:
    if adapter is None:
        return False
    try:
        profile, receipt = adapter.profile, adapter.receipt
        return (isinstance(profile, TargetProfile) and isinstance(receipt, EquivalenceReceipt)
                and profile.digest == profile_sha256 == receipt.profile_sha256
                and profile.adapter_version == receipt.adapter_version
                and profile.converter_version == receipt.converter_version
                and profile.blender_build_hash == receipt.blender_build_hash
                and receipt.verified and source_sha256 in receipt.supported_source_sha256s
                and (not require_patches or receipt.patch_verified))
    except (AttributeError, TypeError, ValueError):
        return False


class CertifiedTargetRegistry:
    """Server-owned adapters only; no client supplied certification flags."""
    def __init__(self):
        self._adapters: dict[str, TargetAdapter] = {}

    def register(self, adapter: TargetAdapter) -> None:
        if not adapter.receipt.verified or adapter.profile.digest != adapter.receipt.profile_sha256:
            raise ValueError("target equivalence unverified")
        if (adapter.profile.adapter_version != adapter.receipt.adapter_version
                or adapter.profile.converter_version != adapter.receipt.converter_version
                or adapter.profile.blender_build_hash != adapter.receipt.blender_build_hash):
            raise ValueError("target adapter version mismatch")
        digest = adapter.profile.digest
        if digest in self._adapters:
            raise ValueError("target profile already registered")
        self._adapters[digest] = adapter

    def resolve(self, profile_sha256: str, source_sha256: str) -> TargetAdapter | None:
        adapter = self._adapters.get(profile_sha256)
        return adapter if certified_adapter(adapter, profile_sha256, source_sha256) else None

    def evaluate(self, profile_sha256: str, source_text: str) -> TargetEvaluation:
        source_sha256 = hashlib.sha256(source_text.encode()).hexdigest()
        adapter = self.resolve(profile_sha256, source_sha256)
        if adapter is None:
            raise ValueError("target_profile_unsupported")
        result = adapter.evaluate(source_text)
        if (not isinstance(result, TargetEvaluation) or result.profile_sha256 != profile_sha256
                or result.source_sha256 != source_sha256
                or result.digest != dict(adapter.receipt.evaluation_sha256s).get(source_sha256)
                or (result.patches and not adapter.receipt.patch_verified)):
            raise ValueError("target_evaluation_identity_mismatch")
        return result


TARGET_REGISTRY = CertifiedTargetRegistry()
