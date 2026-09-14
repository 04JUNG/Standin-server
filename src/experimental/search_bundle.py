"""In-memory foundation shared by opt-in A/B1 search experiments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np

from .a_minimal_support import MinimalSupportGate
from .b1_pose_facts import PoseFactReranker


EXPERIMENTAL_BUNDLE_VERSION = "experimental-search-a-b1-v1"


def _snapshot_id(entries) -> str:
    digest = hashlib.sha256()
    digest.update(EXPERIMENTAL_BUNDLE_VERSION.encode("utf-8"))
    for entry in entries:
        view = entry.view.value if hasattr(entry.view, "value") else str(entry.view)
        digest.update(str(entry.pose_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(view).encode("utf-8"))
        digest.update(b"\0")
        feature = np.ascontiguousarray(entry.feature, dtype=np.float32)
        digest.update(feature.tobytes())
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class ExperimentalSearchBundle:
    """A/B1 indexes tied to the exact immutable geometry entry snapshot."""

    version: str
    snapshot_id: str
    entry_count: int
    a_support_gate: MinimalSupportGate | None
    b1_pose_fact_reranker: PoseFactReranker | None

    @classmethod
    def build(cls, entries: Iterable, *, enable_a: bool,
              enable_b1: bool) -> "ExperimentalSearchBundle":
        frozen_entries = tuple(entries)
        return cls(
            version=EXPERIMENTAL_BUNDLE_VERSION,
            snapshot_id=_snapshot_id(frozen_entries),
            entry_count=len(frozen_entries),
            a_support_gate=(
                MinimalSupportGate.build(frozen_entries) if enable_a else None
            ),
            b1_pose_fact_reranker=(
                PoseFactReranker.build(frozen_entries) if enable_b1 else None
            ),
        )

    def trace_identity(self) -> dict:
        return {
            "bundle_version": self.version,
            "snapshot_id": self.snapshot_id,
            "entry_count": self.entry_count,
        }
