"""ID 없는 군집 집계 — 1년이 지나도 남겨 둘 수 있는 유일한 산출물.

내부 규정상 1년이 넘은 연결 가능 데이터는 지우고 ID 없는 집계만 남길 수 있다.
여기 들어가는 것은 서로 다른 설치 k곳(기본 3) 이상에서 모인 군집의 평균 자세, 건수,
태그 분포뿐이다. 관측·설치 가명은 넣지 않는다. 설치 k곳 미만이 본 관절은 중심 자세에서
비우고, k곳 미만인 태그 값은 합쳐서 숨긴다.

`cluster_key`는 중심 자세를 0.05 격자로 반올림한 값의 해시다. 다음 주기에 같은 자리의
군집이 다시 나오면 키를 유지한 채 이력만 덧붙인다.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import median

import numpy as np

from .config import GapConfig
from .features import BODY, mirror, sym_distance
from .replay import Replayed

AGGREGATES = Path("aggregates") / "clusters.json"
SUPPRESSED = "_other"


def iso_week(day: date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def aligned(item: Replayed, flip: bool) -> tuple[np.ndarray, np.ndarray]:
    return mirror(item.feature, item.mask) if flip else (item.feature, item.mask)


def centroid(members: list[Replayed], flips: list[bool], k: int) -> tuple[list, list]:
    """관절별 평균. 그 관절을 본 설치가 k곳 미만이면 비운다."""
    features, masks = zip(*(aligned(m, f) for m, f in zip(members, flips)))
    points: list = []
    support: list = []
    for joint in range(17):
        seen = [i for i, mask in enumerate(masks) if mask[joint]]
        installations = {members[i].obs.inst for i in seen}
        if len(installations) >= k:
            coords = np.mean([features[i][joint] for i in seen], axis=0)
            points.append([round(float(coords[0]), 4), round(float(coords[1]), 4)])
            support.append(len(installations))
        else:
            points.append(None)
            support.append(None)
    return points, support


def centroid_arrays(points: list) -> tuple[np.ndarray, np.ndarray]:
    feature = np.zeros((17, 2), dtype=np.float32)
    mask = np.zeros(17, dtype=bool)
    for joint, point in enumerate(points):
        if point is not None:
            feature[joint] = point
            mask[joint] = True
    return feature, mask


def cluster_key(coverage_class: str, points: list) -> str:
    grid = ";".join("x" if points[j] is None else f"{round(points[j][0] / 0.05)},{round(points[j][1] / 0.05)}"
                    for j in BODY)
    return "g-" + hashlib.sha256(f"{coverage_class}|{grid}".encode()).hexdigest()[:10]


def histogram(members: list[Replayed], value_of, k: int) -> dict:
    """값별 서로 다른 설치 수. k곳 미만인 값은 `_other`로 합쳐 숫자를 숨긴다."""
    by_value: dict[str, set] = {}
    for member in members:
        value = value_of(member)
        by_value.setdefault("null" if value is None else str(value), set()).add(member.obs.inst)
    out = {value: len(insts) for value, insts in by_value.items() if len(insts) >= k}
    if any(len(insts) < k for insts in by_value.values()):
        out[SUPPRESSED] = f"<{k}"
    return dict(sorted(out.items()))


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(median(values)), 4) if values else None


def build(clusters, items: list[Replayed], cfg: GapConfig, library_version: str | None) -> list[dict]:
    """목표 군집(설치 k곳 이상)만 ID 없는 레코드로 만든다."""
    out = []
    k = cfg.k_min_installations
    for cluster in clusters:
        if cluster.installations < k:
            continue
        members = [items[i] for i in cluster.members]
        points, support = centroid(members, cluster.mirrored, k)
        days = [m.obs.observed_on for m in members]
        week = iso_week(max(days))
        filled = sum(1 for m in members if m.label == "filled_pending_deploy")
        families = histogram(members, lambda m: m.prod_pose, k)
        out.append({
            "cluster_key": cluster_key(cluster.coverage_class, points),
            "coverage_class": cluster.coverage_class,
            "thresholds_version": cfg.version,
            "first_week": iso_week(min(days)),
            "last_week": week,
            "support": {"installations": cluster.installations, "observations": len(members)},
            "centroid": points,
            "centroid_joint_support": support,
            "histograms": {
                "person_action": histogram(members, lambda m: m.obs.tags.get("action"), k),
                "person_view": histogram(members, lambda m: m.obs.tags.get("view"), k),
                "cut_shot": histogram(members, lambda m: m.obs.cut.get("tags", {}).get("shot"), k),
                "scope": histogram(members, lambda m: m.obs.scope.get("detected"), k),
                "feedback": histogram(members, lambda m: m.obs.behavior.get("job_feedback"), k),
                "production_top1": families,
            },
            "gap": {
                "prod_top1_median": _median(m.prod_distance for m in members),
                "curated_top1_median": _median(m.cur_distance for m in members),
                "strong_share": round(sum(m.severity == "strong" for m in members) / len(members), 3),
            },
            "status": "filled_pending_deploy" if filled * 2 > len(members) else "open",
            "history": [{
                "week": week,
                "library_version": library_version,
                "installations": cluster.installations,
                "prod_top1_median": _median(m.prod_distance for m in members),
            }],
        })
    return out


def match(existing: list[dict], record: dict, eps: float) -> dict | None:
    feature, mask = centroid_arrays(record["centroid"])
    best, best_d = None, float("inf")
    for old in existing:
        if old["coverage_class"] != record["coverage_class"]:
            continue
        old_f, old_m = centroid_arrays(old["centroid"])
        distance, _ = sym_distance(feature, mask, old_f, old_m, min_common=6)
        if distance <= eps / 2 and distance < best_d:
            best, best_d = old, distance
    return best


def merge(existing: list[dict], new: list[dict], eps: float) -> list[dict]:
    """같은 자리의 군집이면 키·첫 주를 유지하고 최신 값으로 갱신, 이력은 덧붙인다."""
    merged = [dict(record) for record in existing]
    for record in new:
        old = match(merged, record, eps)
        if old is None:
            merged.append(record)
            continue
        old["last_week"] = max(old["last_week"], record["last_week"])
        old["first_week"] = min(old["first_week"], record["first_week"])
        for field in ("support", "centroid", "centroid_joint_support", "histograms", "gap",
                      "thresholds_version"):
            old[field] = record[field]
        if old.get("status") not in ("in_progress", "closed"):
            old["status"] = record["status"]
        old["history"] = list(old.get("history", [])) + record["history"]
    return sorted(merged, key=lambda r: (-r["support"]["installations"], r["cluster_key"]))


def measure(aggregates: list[dict], items: list[Replayed], cfg: GapConfig,
            library_version: str | None, new_pose_ids: set[str], today: date) -> list[dict]:
    """배포 뒤 관측을 목표 군집에 대응시켜 메움률·신규 포즈 Top-1 비중을 이력에 남긴다."""
    k = cfg.k_min_installations
    eligible = [i for i in items if i.excluded is None and i.prod_distance is not None]
    for record in aggregates:
        feature, mask = centroid_arrays(record["centroid"])
        matched = [i for i in eligible
                   if sym_distance(feature, mask, i.feature, i.mask, min_common=6)[0] <= cfg.eps_cluster]
        installations = len({i.obs.inst for i in matched})
        row = {"week": iso_week(today), "library_version": library_version, "measured": True}
        if installations >= k:
            fill = sum(i.prod_distance <= cfg.tau_fill for i in matched) / len(matched)
            new = sum(i.prod_pose in new_pose_ids for i in matched) / len(matched)
            row.update(installations=installations, fill_rate=round(fill, 3),
                       new_pose_top1_share=round(new, 3),
                       prod_top1_median=_median(i.prod_distance for i in matched))
            if fill >= 0.5:
                record["status"] = "closed"
        else:
            row["installations"] = f"<{k}"
        record["history"] = list(record.get("history", [])) + [row]
    return aggregates


def load(root: Path) -> list[dict]:
    path = Path(root) / AGGREGATES
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["clusters"]


def save(root: Path, aggregates: list[dict]) -> Path:
    path = Path(root) / AGGREGATES
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "clusters": aggregates},
                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


# BFF export 가명 모양: "o_" / "i_" + base64url 22자.
_PSEUDONYM = re.compile(r'"[oi]_[A-Za-z0-9_-]{22}"')


def assert_id_free(aggregates: list[dict]) -> None:
    """집계에 관측·설치 가명이 섞이지 않았는지 확인한다(저장 직전 호출)."""
    text = json.dumps(aggregates, ensure_ascii=False)
    if '"obs"' in text or '"inst"' in text or _PSEUDONYM.search(text):
        raise ValueError("aggregate contains an observation-level identifier")
