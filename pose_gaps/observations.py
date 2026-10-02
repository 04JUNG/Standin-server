"""BFF 공백 관측 export(schema 1)를 읽는 쪽의 계약.

BFF `GET /v1/admin/gaps/observations`가 내보내는 항목 하나가 `Observation` 하나다.
작업·설치 ID, 입력 해시, S3 key, bbox, 이미지 크기는 export에 없어야 한다. 여기서도
한 번 더 막는다 — 하나라도 섞여 오면 그 스냅샷 전체를 받지 않는다.

`obs`·`inst`는 export마다 새 salt로 만든 가명이라 같은 export 안에서만 의미가 있다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from . import EXPORT_SCHEMA_VERSION

FORBIDDEN_KEYS = frozenset({
    "job_id", "jobId", "installation_id", "installationId", "input_sha256", "inputSha256",
    "input_s3_key", "inputS3Key", "s3_key", "bbox", "bbox_xyxy", "image_width", "image_height",
})
# BFF가 발급하는 원본 ID 모양. 가명(o_…, i_…)은 이 모양이 아니다.
RAW_ID_PATTERNS = (
    re.compile(r"inst_[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE),
    re.compile(r"job_[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE),
    re.compile(r"user_[0-9a-f]{12}"),
    re.compile(r"installations/"),
)


class ObservationError(ValueError):
    """export 항목이 계약과 다르다."""


@dataclass(frozen=True)
class Candidate:
    pose_id: str
    view: str
    rank: int
    distance: float | None
    match_level: str | None = None


@dataclass(frozen=True)
class Observation:
    obs: str
    inst: str
    observed_on: date
    expires_on: date
    keypoints: np.ndarray
    raw_scores: np.ndarray | None
    effective_scores: np.ndarray | None
    evidence_mask: np.ndarray | None
    search_mask: np.ndarray | None
    coverage_class: str | None
    skeleton_state: str | None
    skeleton_source: str | None
    slot_origin: str | None
    confidence: str | None
    search_scope: str | None
    distance_metric: str | None
    search_stability: str | None
    rank_distance: float | None
    confidence_threshold: float | None
    quality_reasons: tuple[str, ...] = ()
    tags: dict = field(default_factory=dict)
    scope: dict = field(default_factory=dict)
    cut: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    candidates: tuple[Candidate, ...] = ()
    behavior: dict = field(default_factory=dict)

    @property
    def library_version(self) -> str | None:
        value = self.versions.get("pose_library")
        return str(value) if value is not None else None


def _walk(value: Any, path: str = ""):
    """모든 노드를 (위치, 키 또는 None, 값)으로 돌려준다. 리스트 안의 문자열도 포함한다."""
    if isinstance(value, dict):
        for key, item in value.items():
            where = f"{path}.{key}"
            yield where, key, item
            yield from _walk(item, where)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            where = f"{path}[{index}]"
            yield where, None, item
            yield from _walk(item, where)


def privacy_problems(raw: dict) -> list[str]:
    """금지 키와 원본 ID 모양 문자열을 찾는다. 값 자체는 돌려주지 않는다."""
    problems = []
    for where, key, item in _walk(raw):
        if key is not None and key in FORBIDDEN_KEYS:
            problems.append(f"forbidden key {where}")
        texts = [text for text in (key, item) if isinstance(text, str)]
        if any(pattern.search(text) for text in texts for pattern in RAW_ID_PATTERNS):
            problems.append(f"raw identifier at {where}")
    return problems


def _array17(value, *, dtype, name: str, required: bool = False):
    if value is None or value == []:
        if required:
            raise ObservationError(f"{name} is required")
        return None
    array = np.asarray(value, dtype=dtype)
    if array.shape[0] != 17:
        raise ObservationError(f"{name} must have 17 rows, got {array.shape}")
    if dtype is not bool and not np.isfinite(array).all():
        raise ObservationError(f"{name} contains NaN/Inf")
    return array


def _optional_float(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ObservationError("distance values must be finite")
    return number


def parse_observation(raw: dict) -> Observation:
    if not isinstance(raw, dict):
        raise ObservationError("observation must be an object")
    problems = privacy_problems(raw)
    if problems:
        raise ObservationError("privacy: " + "; ".join(sorted(set(problems))[:5]))
    person = raw.get("person") or {}
    keypoints = _array17(person.get("keypoints"), dtype=np.float32, name="person.keypoints",
                         required=True)
    if keypoints.shape != (17, 2):
        raise ObservationError(f"person.keypoints must be (17, 2), got {keypoints.shape}")
    candidates = tuple(
        Candidate(pose_id=str(c["pose_id"]), view=str(c["view"]), rank=int(c["rank"]),
                  distance=_optional_float(c.get("distance")), match_level=c.get("match_level"))
        for c in sorted(raw.get("candidates") or [], key=lambda c: int(c["rank"])))
    try:
        observed_on = date.fromisoformat(str(raw["observed_on"]))
        expires_on = date.fromisoformat(str(raw["expires_on"]))
        obs, inst = str(raw["obs"]), str(raw["inst"])
    except (KeyError, ValueError) as exc:
        raise ObservationError(f"obs/inst/observed_on/expires_on: {exc}") from exc
    return Observation(
        obs=obs, inst=inst, observed_on=observed_on, expires_on=expires_on,
        keypoints=keypoints,
        raw_scores=_array17(person.get("raw_scores"), dtype=np.float32, name="raw_scores"),
        effective_scores=_array17(person.get("effective_scores"), dtype=np.float32,
                                  name="effective_scores"),
        evidence_mask=_array17(person.get("evidence_mask"), dtype=bool, name="evidence_mask"),
        search_mask=_array17(person.get("search_mask"), dtype=bool, name="search_mask"),
        coverage_class=person.get("coverage_class"),
        skeleton_state=person.get("skeleton_state"),
        skeleton_source=person.get("skeleton_source"),
        slot_origin=person.get("slot_origin"),
        confidence=person.get("confidence"),
        search_scope=person.get("search_scope"),
        distance_metric=person.get("distance_metric"),
        search_stability=person.get("search_stability"),
        rank_distance=_optional_float(person.get("rank_distance")),
        confidence_threshold=_optional_float(person.get("confidence_threshold")),
        quality_reasons=tuple(str(r) for r in person.get("quality_reasons") or ()),
        tags=dict(person.get("tags") or {}),
        scope=dict(person.get("scope") or {}),
        cut=dict(raw.get("cut") or {}),
        versions=dict(raw.get("versions") or {}),
        candidates=candidates,
        behavior=dict(raw.get("behavior") or {}),
    )


def check_page(page: dict) -> list[dict]:
    """export 응답 한 페이지의 머리를 확인하고 항목 원본(dict)을 돌려준다."""
    if page.get("schemaVersion") != EXPORT_SCHEMA_VERSION:
        raise ObservationError(f"unsupported export schemaVersion {page.get('schemaVersion')!r}")
    items = page.get("items")
    if not isinstance(items, list):
        raise ObservationError("export page items must be a list")
    return items


def load_jsonl(path: Path) -> list[Observation]:
    out = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                out.append(parse_observation(json.loads(line)))
            except (ObservationError, ValueError, KeyError) as exc:
                raise ObservationError(f"{path}:{line_no}: {exc}") from exc
    return out


def dump_jsonl(path: Path, items: Iterable[dict]) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count
