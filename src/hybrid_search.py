"""독립 실행 가능한 vectorized hybrid 포즈 검색 실험 모듈.

기존 ``src.search``의 운영 경로를 수정하지 않고 다음을 비교한다.

``position``
    현재 관절 위치 평균 L2 거리의 vectorized 구현.
``angle``
    현재 뼈 방향 cosine 거리의 vectorized 구현.
``hybrid_h0``
    현재 raw hybrid, ``(1-w) * position + w * angle``.
``hybrid_h1``
    coverage별 고정 scale로 두 성분을 보정한 hybrid.
``hybrid_h2``
    H1에 짧게 투영된 뼈의 방향 신뢰도 감쇠를 추가한 hybrid.

검색은 position shortlist를 만들지 않고 모든 projection을 한 번에 계산한 뒤
stable sort한다. 결과는 pose family마다 최선의 variant 하나만 남긴다.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from .features import _BODY, _BONES
from .schema import LibraryEntry, PoseCandidate


METRICS = frozenset({
    "position", "angle", "hybrid_h0", "hybrid_h1", "hybrid_h2",
})
_EPS = 1e-6


def pose_family_id(pose_id: str, meta: dict | None = None) -> str:
    """명시적 family ID를 우선하고, 없으면 ``_mirror`` suffix를 접는다."""
    if meta and meta.get("pose_family_id"):
        return str(meta["pose_family_id"])
    suffix = "_mirror"
    return pose_id[:-len(suffix)] if pose_id.lower().endswith(suffix) else pose_id


def _joint_mask(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.ones(17, dtype=bool)
    out = np.asarray(mask, dtype=bool).reshape(-1)
    if out.shape != (17,):
        raise ValueError(f"joint mask must have shape (17,), got {out.shape}")
    return out


def _bone_mask(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.ones(len(_BONES), dtype=bool)
    out = np.asarray(mask, dtype=bool).reshape(-1)
    if out.shape != (len(_BONES),):
        raise ValueError(
            f"observable bones must have shape ({len(_BONES)},), got {out.shape}"
        )
    return out


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array)
    out.flags.writeable = False
    return out


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """``low`` 이하는 0, ``high`` 이상은 1인 연속 신뢰도 함수."""
    if not np.isfinite([low, high]).all() or low < 0 or high <= low:
        raise ValueError("short-bone thresholds require 0 <= low < high")
    x = np.clip((values - low) / (high - low), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class HybridScaleProfile:
    """한 coverage class에서 고정할 component scale과 angle 비중."""

    position_scale: float = 1.0
    angle_scale: float = 1.0
    angle_weight: float = 0.7

    def __post_init__(self) -> None:
        if not np.isfinite([self.position_scale, self.angle_scale]).all():
            raise ValueError("hybrid scales must be finite")
        if self.position_scale <= 0 or self.angle_scale <= 0:
            raise ValueError("hybrid scales must be positive")
        if not 0.0 <= self.angle_weight <= 1.0:
            raise ValueError("angle_weight must be in [0, 1]")


@dataclass(frozen=True)
class HybridMetricConfig:
    """실험 metric 설정. Scale 기본값 1은 H0와 비교 가능한 안전한 시작점이다."""

    metric: str = "hybrid_h0"
    full: HybridScaleProfile = HybridScaleProfile()
    reduced: HybridScaleProfile = HybridScaleProfile()
    short_bone_low: float = 0.05
    short_bone_high: float = 0.20
    min_effective_bone_mass: float = 4.0
    missing_observation_penalty: float = 0.0
    metric_version: str = "hybrid-lab-v1"

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}")
        if self.min_effective_bone_mass < 0:
            raise ValueError("min_effective_bone_mass must be >= 0")
        if self.missing_observation_penalty < 0:
            raise ValueError("missing_observation_penalty must be >= 0")
        _smoothstep(
            np.asarray([self.short_bone_low], dtype=np.float32),
            self.short_bone_low,
            self.short_bone_high,
        )

    def profile(self, coverage_class: str) -> HybridScaleProfile:
        if coverage_class == "full":
            return self.full
        if coverage_class in {"reduced", "sparse"}:
            return self.reduced
        raise ValueError(
            "coverage_class must be full, reduced, or sparse for search"
        )


@dataclass(frozen=True)
class BatchComponents:
    position: np.ndarray
    angle: np.ndarray
    observed_angle: np.ndarray
    common_bone_count: np.ndarray
    effective_bone_mass: np.ndarray
    missing_bone_fraction: np.ndarray


@dataclass(frozen=True)
class HybridSearchHit:
    candidate: PoseCandidate
    row_index: int
    position_distance: float
    angle_distance: float
    observed_angle_distance: float
    common_bone_count: int
    effective_bone_mass: float
    position_row_rank: int
    angle_row_rank: int


@dataclass(frozen=True)
class HybridSearchResult:
    hits: tuple[HybridSearchHit, ...]
    elapsed_ms: float
    metric: str
    metric_version: str
    coverage_class: str
    projection_count: int
    eligible_projection_count: int

    @property
    def candidates(self) -> tuple[PoseCandidate, ...]:
        return tuple(hit.candidate for hit in self.hits)


@dataclass(frozen=True)
class GeometricSearchIndex:
    """검색 시 재사용하는 immutable projection 배열."""

    entries: tuple[LibraryEntry, ...]
    features: np.ndarray
    bone_dirs: np.ndarray
    bone_lengths: np.ndarray
    bone_valid: np.ndarray
    pose_ids: tuple[str, ...]
    family_ids: tuple[str, ...]

    @classmethod
    def build(cls, entries: Iterable[LibraryEntry]) -> "GeometricSearchIndex":
        frozen_entries = tuple(entries)
        if frozen_entries:
            features = np.stack([
                np.asarray(entry.feature, dtype=np.float32).reshape(17, 2)
                for entry in frozen_entries
            ])
            if not np.isfinite(features).all():
                raise ValueError("library features contain NaN/Inf")
        else:
            features = np.empty((0, 17, 2), dtype=np.float32)

        bone_pairs = np.asarray(_BONES, dtype=np.intp)
        vectors = features[:, bone_pairs[:, 1]] - features[:, bone_pairs[:, 0]]
        lengths = np.linalg.norm(vectors, axis=2)
        valid = lengths > _EPS
        dirs = np.zeros_like(vectors, dtype=np.float32)
        np.divide(vectors, lengths[..., None], out=dirs,
                  where=valid[..., None])

        return cls(
            entries=frozen_entries,
            features=_readonly(features.astype(np.float32, copy=False)),
            bone_dirs=_readonly(dirs),
            bone_lengths=_readonly(lengths.astype(np.float32, copy=False)),
            bone_valid=_readonly(valid),
            pose_ids=tuple(entry.pose_id for entry in frozen_entries),
            family_ids=tuple(
                pose_family_id(entry.pose_id, entry.meta)
                for entry in frozen_entries
            ),
        )

    @property
    def memory_bytes(self) -> int:
        return int(
            self.features.nbytes + self.bone_dirs.nbytes
            + self.bone_lengths.nbytes + self.bone_valid.nbytes
        )

    def batch_components(
        self,
        query: np.ndarray,
        query_valid_mask: np.ndarray | None = None,
        query_observable_bones: np.ndarray | None = None,
        *,
        short_bone_low: float = 0.05,
        short_bone_high: float = 0.20,
        missing_observation_penalty: float = 0.0,
    ) -> BatchComponents:
        """전체 projection의 position/raw-angle/observed-angle을 한 번에 계산한다."""
        query_points = np.asarray(query, dtype=np.float32).reshape(17, 2)
        if not np.isfinite(query_points).all():
            raise ValueError("query feature contains NaN/Inf")
        joint_valid = _joint_mask(query_valid_mask)
        observable = _bone_mask(query_observable_bones)
        n = len(self.entries)

        body = np.asarray(_BODY, dtype=np.intp)
        body_valid = joint_valid[body]
        if body_valid.any():
            delta = self.features[:, body[body_valid]] - query_points[body[body_valid]]
            position = np.linalg.norm(delta, axis=2).mean(axis=1)
        else:
            position = np.full(n, np.inf, dtype=np.float32)

        bone_pairs = np.asarray(_BONES, dtype=np.intp)
        query_vectors = (
            query_points[bone_pairs[:, 1]] - query_points[bone_pairs[:, 0]]
        )
        query_lengths = np.linalg.norm(query_vectors, axis=1)
        query_bone_valid = (
            joint_valid[bone_pairs[:, 0]]
            & joint_valid[bone_pairs[:, 1]]
            & observable
            & (query_lengths > _EPS)
        )
        query_dirs = np.zeros_like(query_vectors, dtype=np.float32)
        np.divide(query_vectors, query_lengths[:, None], out=query_dirs,
                  where=query_bone_valid[:, None])

        common = self.bone_valid & query_bone_valid[None, :]
        common_count = common.sum(axis=1, dtype=np.int32)
        cosine = np.clip(
            np.einsum("nbi,bi->nb", self.bone_dirs, query_dirs),
            -1.0,
            1.0,
        )
        raw_terms = (1.0 - cosine) * common
        angle = np.full(n, 2.0, dtype=np.float32)
        np.divide(raw_terms.sum(axis=1), common_count, out=angle,
                  where=common_count > 0)

        query_reliability = _smoothstep(
            query_lengths, short_bone_low, short_bone_high
        ) * query_bone_valid
        library_reliability = _smoothstep(
            self.bone_lengths, short_bone_low, short_bone_high
        ) * self.bone_valid
        reliability = library_reliability * query_reliability[None, :]
        effective_mass = reliability.sum(axis=1)
        observed_angle = np.full(n, 2.0, dtype=np.float32)
        np.divide(
            (raw_terms * reliability).sum(axis=1),
            effective_mass,
            out=observed_angle,
            where=effective_mass > _EPS,
        )

        query_bone_count = int(query_bone_valid.sum())
        if query_bone_count:
            missing = query_bone_valid[None, :] & ~self.bone_valid
            missing_fraction = missing.sum(axis=1) / float(query_bone_count)
        else:
            missing_fraction = np.ones(n, dtype=np.float32)
        if missing_observation_penalty:
            observed_angle = (
                observed_angle
                + float(missing_observation_penalty) * missing_fraction
            )

        return BatchComponents(
            position=_readonly(np.asarray(position, dtype=np.float32)),
            angle=_readonly(np.asarray(angle, dtype=np.float32)),
            observed_angle=_readonly(np.asarray(observed_angle, dtype=np.float32)),
            common_bone_count=_readonly(np.asarray(common_count, dtype=np.int32)),
            effective_bone_mass=_readonly(
                np.asarray(effective_mass, dtype=np.float32)
            ),
            missing_bone_fraction=_readonly(
                np.asarray(missing_fraction, dtype=np.float32)
            ),
        )

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 5,
        config: HybridMetricConfig | None = None,
        coverage_class: str = "full",
        query_valid_mask: np.ndarray | None = None,
        query_observable_bones: np.ndarray | None = None,
        quarantined_pose_ids: Sequence[str] = (),
    ) -> HybridSearchResult:
        """Exhaustive vectorized 검색 후 family별 하나를 반환한다."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        config = config or HybridMetricConfig()
        profile = config.profile(coverage_class)
        started = perf_counter()
        components = self.batch_components(
            query,
            query_valid_mask,
            query_observable_bones,
            short_bone_low=config.short_bone_low,
            short_bone_high=config.short_bone_high,
            missing_observation_penalty=config.missing_observation_penalty,
        )

        if config.metric == "position":
            distances = components.position
        elif config.metric == "angle":
            distances = components.angle
        elif config.metric == "hybrid_h0":
            distances = (
                (1.0 - profile.angle_weight) * components.position
                + profile.angle_weight * components.angle
            )
        else:
            normalized_position = components.position / profile.position_scale
            chosen_angle = (
                components.observed_angle
                if config.metric == "hybrid_h2" else components.angle
            )
            normalized_angle = chosen_angle / profile.angle_scale
            if config.metric == "hybrid_h2" and config.min_effective_bone_mass > 0:
                mass_ratio = np.clip(
                    components.effective_bone_mass
                    / config.min_effective_bone_mass,
                    0.0,
                    1.0,
                )
                angle_weight = profile.angle_weight * mass_ratio
            else:
                angle_weight = profile.angle_weight
            distances = (
                (1.0 - angle_weight) * normalized_position
                + angle_weight * normalized_angle
            )

        distances = np.asarray(distances, dtype=np.float32)
        order = np.argsort(distances, kind="stable")
        position_order = np.argsort(components.position, kind="stable")
        angle_order = np.argsort(components.angle, kind="stable")
        position_ranks = np.empty(len(self.entries), dtype=np.int64)
        angle_ranks = np.empty(len(self.entries), dtype=np.int64)
        position_ranks[position_order] = np.arange(len(self.entries))
        angle_ranks[angle_order] = np.arange(len(self.entries))

        quarantine = frozenset(str(value) for value in quarantined_pose_ids)
        eligible_count = sum(pose_id not in quarantine for pose_id in self.pose_ids)
        seen_families: set[str] = set()
        hits: list[HybridSearchHit] = []
        for row in order:
            row_index = int(row)
            entry = self.entries[row_index]
            if entry.pose_id in quarantine:
                continue
            family_id = self.family_ids[row_index]
            if family_id in seen_families:
                continue
            seen_families.add(family_id)
            candidate = PoseCandidate(
                pose_id=entry.pose_id,
                view=entry.view,
                distance=float(distances[row_index]),
                tags=entry.tags,
                bvh_path=entry.bvh_path,
                pose_family_id=family_id,
            )
            hits.append(HybridSearchHit(
                candidate=candidate,
                row_index=row_index,
                position_distance=float(components.position[row_index]),
                angle_distance=float(components.angle[row_index]),
                observed_angle_distance=float(
                    components.observed_angle[row_index]
                ),
                common_bone_count=int(
                    components.common_bone_count[row_index]
                ),
                effective_bone_mass=float(
                    components.effective_bone_mass[row_index]
                ),
                position_row_rank=int(position_ranks[row_index]) + 1,
                angle_row_rank=int(angle_ranks[row_index]) + 1,
            ))
            if len(hits) >= top_k:
                break

        elapsed_ms = (perf_counter() - started) * 1000.0
        return HybridSearchResult(
            hits=tuple(hits),
            elapsed_ms=elapsed_ms,
            metric=config.metric,
            metric_version=config.metric_version,
            coverage_class=coverage_class,
            projection_count=len(self.entries),
            eligible_projection_count=eligible_count,
        )


def family_overlap(
    first: HybridSearchResult,
    second: HybridSearchResult,
) -> int:
    """두 결과의 Top-K family 교집합 크기."""
    a = {hit.candidate.pose_family_id for hit in first.hits}
    b = {hit.candidate.pose_family_id for hit in second.hits}
    return len(a & b)
