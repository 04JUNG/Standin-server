"""관측을 원본 이미지 없이 다시 검색하고 공백 라벨을 붙인다.

먼저 충실도를 본다: 같은 라이브러리 버전에서 나온 관측은 재계산한 Top-1이 export의
`rank_distance`·Top-1 pose와 같아야 한다. 어긋나면 정규화·마스크·quarantine·metric 중
무언가가 운영과 다르다는 뜻이고, 그 상태로 낸 공백 목록은 믿을 수 없으니 멈춘다.

라벨(운영 라이브러리 거리 d_prod, 정리 라이브러리 거리 d_cur):
- not_gap: d_prod ≤ τ_soft
- extraction_suspect: d_prod ≥ extraction_cap이고 "후보가 엉뚱함" 피드백이 없음(추출 실패 쪽)
- filled_pending_deploy: d_cur ≤ τ_fill — 정리 라이브러리에 이미 있고 배포만 남았다
- gap_open: 나머지. severity는 d_prod > τ_strong이면 strong, 아니면 soft
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GapConfig
from .eligibility import ineligibility
from .features import query_feature
from .libraries import Library
from .observations import Observation


class FidelityError(RuntimeError):
    """재계산한 운영 검색이 export와 다르다."""


@dataclass
class Replayed:
    obs: Observation
    feature: np.ndarray
    mask: np.ndarray
    prod_pose: str | None
    prod_distance: float | None
    cur_pose: str | None = None
    cur_distance: float | None = None
    label: str = "not_gap"
    severity: str | None = None
    excluded: str | None = None
    extra: dict = field(default_factory=dict)


def _metric(obs: Observation) -> str:
    return (obs.distance_metric or "pos").lower()


def check_fidelity(observations: list[Observation], library: Library, cfg: GapConfig) -> dict:
    """라이브러리와 같은 버전의 관측만 비교한다. 비교 대상이 없으면 checked=0."""
    same = [o for o in observations
            if library.version is not None and o.library_version == library.version
            and o.rank_distance is not None and o.candidates and ineligibility(o, cfg.kpt_threshold) is None]
    matched = 0
    for obs in same:
        feature, mask = query_feature(obs, cfg.kpt_threshold)
        hit = library.top1(feature, mask, metric=_metric(obs))
        if (hit is not None and hit.pose_id == obs.candidates[0].pose_id
                and abs(hit.distance - obs.rank_distance) <= cfg.fidelity_tolerance):
            matched += 1
    share = matched / len(same) if same else None
    result = {"library_version": library.version, "checked": len(same), "matched": matched,
              "share": share, "required": cfg.fidelity_min_share}
    if same and share < cfg.fidelity_min_share:
        raise FidelityError(
            f"재검색이 운영 결과와 {matched}/{len(same)}만 일치합니다(기준 {cfg.fidelity_min_share:.0%}). "
            "정규화·마스크·quarantine·metric이 운영과 같은지 확인하세요")
    return result


def label(prod_distance: float, cur_distance: float | None, obs: Observation,
          cfg: GapConfig) -> tuple[str, str | None]:
    if prod_distance <= cfg.tau_soft:
        return "not_gap", None
    if (prod_distance >= cfg.extraction_cap
            and obs.behavior.get("job_feedback") != "candidates_irrelevant"):
        return "extraction_suspect", None
    if cur_distance is not None and cur_distance <= cfg.tau_fill:
        return "filled_pending_deploy", None
    return "gap_open", ("strong" if prod_distance > cfg.tau_strong else "soft")


def replay(observations: list[Observation], production: Library, cfg: GapConfig,
           curated: Library | None = None) -> list[Replayed]:
    """적격 관측을 현재 운영 라이브러리(와 정리 라이브러리)로 다시 검색해 라벨을 붙인다."""
    out = []
    for obs in observations:
        reason = ineligibility(obs, cfg.kpt_threshold)
        if reason is not None:
            out.append(Replayed(obs, np.zeros((17, 2), np.float32), np.zeros(17, bool),
                                None, None, label="excluded", excluded=reason))
            continue
        feature, mask = query_feature(obs, cfg.kpt_threshold)
        prod = production.top1(feature, mask, metric=_metric(obs))
        if prod is None:
            out.append(Replayed(obs, feature, mask, None, None, label="excluded",
                                excluded="no_library_hit"))
            continue
        item = Replayed(obs, feature, mask, prod.pose_id, float(prod.distance))
        if curated is not None:
            cur = curated.top1(feature, mask, metric=_metric(obs))
            if cur is not None:
                item.cur_pose, item.cur_distance = cur.pose_id, float(cur.distance)
        item.label, item.severity = label(item.prod_distance, item.cur_distance, obs, cfg)
        out.append(item)
    return out


def summarize(items: list[Replayed]) -> dict:
    labels: dict[str, int] = {}
    excluded: dict[str, int] = {}
    for item in items:
        labels[item.label] = labels.get(item.label, 0) + 1
        if item.excluded:
            excluded[item.excluded] = excluded.get(item.excluded, 0) + 1
    gaps = [i for i in items if i.label == "gap_open"]
    return {
        "observations": len(items),
        "labels": dict(sorted(labels.items())),
        "excluded": dict(sorted(excluded.items())),
        "gap_installations": len({i.obs.inst for i in gaps}),
        "gap_strong": sum(1 for i in gaps if i.severity == "strong"),
    }
