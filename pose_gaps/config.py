"""`config/pose_gaps.json` 임계값. 값이 바뀌면 `version`을 올린다 — 보고서·집계에 같이 남는다."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pose_gaps.json"


@dataclass(frozen=True)
class GateThresholds:
    improve_delta: float = -0.02
    regress_delta: float = 0.02
    max_regressed_share: float = 0.02
    high_edge: float = 0.25
    high_slack: float = 0.05
    max_family_share_floor: float = 0.05
    max_family_share_multiplier: float = 3.0


@dataclass(frozen=True)
class GapConfig:
    version: str
    tau_soft: float = 0.25
    tau_strong: float = 0.35
    tau_fill: float = 0.20
    extraction_cap: float = 0.60
    eps_cluster: float = 0.22
    k_min_installations: int = 3
    min_common_joints: int = 10
    kpt_threshold: float = 0.3
    fidelity_min_share: float = 0.99
    fidelity_tolerance: float = 1e-4
    ttl_days: int = 14
    max_snapshot_age_days: int = 7
    gate: GateThresholds = field(default_factory=GateThresholds)

    def __post_init__(self):
        if not self.version:
            raise ValueError("pose_gaps config version is required")
        if not 0 < self.tau_fill <= self.tau_soft <= self.tau_strong < self.extraction_cap:
            raise ValueError("thresholds must satisfy 0 < tau_fill <= tau_soft <= tau_strong < extraction_cap")
        if self.k_min_installations < 3:
            raise ValueError("k_min_installations below 3 would keep re-identifiable aggregates")
        if not 0 < self.max_snapshot_age_days <= self.ttl_days:
            raise ValueError("max_snapshot_age_days must be within ttl_days")
        if not 4 <= self.min_common_joints <= 12:
            raise ValueError("min_common_joints must be between 4 and 12 body joints")


def load_config(path: Path | None = None) -> GapConfig:
    raw = json.loads(Path(path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    raw.pop("note", None)
    gate = GateThresholds(**raw.pop("gate", {}))
    return GapConfig(gate=gate, **raw)
