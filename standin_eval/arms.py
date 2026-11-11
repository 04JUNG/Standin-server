"""Explicit experiment arms; historical names remain historical identities."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .util import hash_json

EXTENSION_SCHEMA = "evaluation-extension-v1"
STATES = frozenset({"pass", "fail", "unknown", "unsupported", "missing_dependency",
                    "timeout", "error", "abstained"})


@dataclass(frozen=True)
class Arm:
    arm_id: str
    algorithm: str
    adapter: str
    mode: str | None = None
    expected_version_prefix: str | None = None
    requested_backend: str | None = None
    bindings: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ArmRegistry:
    arms: tuple[Arm, ...]
    baseline: str

    def __post_init__(self):
        ids = [arm.arm_id for arm in self.arms]
        if not ids or len(ids) != len(set(ids)) or self.baseline not in ids:
            raise ValueError("unique nonempty arms and an explicit registered baseline are required")
        if any(not arm.arm_id or not arm.algorithm or not arm.adapter for arm in self.arms):
            raise ValueError("arm identity/algorithm/adapter cannot be empty")

    def to_dict(self) -> dict:
        return {"schema_version": EXTENSION_SCHEMA, "baseline": self.baseline,
                "arms": [arm.to_dict() for arm in self.arms]}

    @property
    def sha256(self) -> str:
        return hash_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict) -> "ArmRegistry":
        return cls(tuple(Arm(**row) for row in value["arms"]), value["baseline"])


# F25 is deliberately not an alias for the old v2.4 B2 arm. V3 currently has
# no applied-output HTTP contract; its shadow calculation is not an output arm.
DEFAULT_REGISTRY = ArmRegistry((
    Arm("B0_no_refine", "selected_source", "identity", requested_backend="identity"),
    Arm("F25", "refine_v2.5", "http", "aggressive", "v2.5", "numpy"),
    Arm("v3", "refine_v3_regional", "http_v3", requested_backend="numpy"),
), baseline="F25")
LEGACY_ARMS = {
    "B0_no_refine": DEFAULT_REGISTRY.arms[0],
    "B1_v1": Arm("B1_v1", "refine_v1", "http", "conservative", "v1", "numpy"),
    "B2_v24_aggressive": Arm("B2_v24_aggressive", "refine_v2.4", "http", "aggressive", "v2.4", "numpy"),
}
