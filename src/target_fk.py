"""Exact, finite Converter-snapshot adapter; no approximate bone-length FK.

This provides bounded CPU evaluation for independently certified, already
converted source poses only. It deliberately abstains for unseen rotations.
Continuous fast FK/shape/patch equivalence remains unsupported in production.
"""
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
from src.target_profile import (TargetEvaluation, TargetProfile, EquivalenceReceipt,
                                certified_adapter)


@dataclass(frozen=True, init=False)
class SnapshotTargetAdapter:
    profile: TargetProfile
    receipt: EquivalenceReceipt
    _snapshots: object
    coordinate_frame = "right-handed-z-up"

    def __init__(self, profile: TargetProfile, receipt: EquivalenceReceipt,
                 evaluations: tuple[TargetEvaluation, ...]):
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "receipt", receipt)
        if not isinstance(evaluations, tuple):
            raise ValueError("snapshot evaluations must be immutable")
        snapshots = {evaluation.source_sha256: evaluation for evaluation in evaluations}
        if (len(snapshots) != len(evaluations)
                or set(snapshots) != set(receipt.supported_source_sha256s)):
            raise ValueError("snapshot certificate source mismatch")
        for source_sha, evaluation in snapshots.items():
            if (not certified_adapter(self, profile.digest, source_sha)
                    or evaluation.profile_sha256 != profile.digest
                    or evaluation.digest != dict(receipt.evaluation_sha256s).get(source_sha)
                    or (evaluation.patches and not receipt.patch_verified)):
                raise ValueError("snapshot equivalence unverified")
        object.__setattr__(self, "_snapshots", MappingProxyType(snapshots))

    def evaluate(self, source_text: str) -> TargetEvaluation:
        source_sha = hashlib.sha256(source_text.encode()).hexdigest()
        if not certified_adapter(self, self.profile.digest, source_sha):
            raise ValueError("target_source_outside_certified_envelope")
        return self._snapshots[source_sha]
