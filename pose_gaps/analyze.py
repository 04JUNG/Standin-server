"""스냅샷 하나를 분석한다: 충실도 → 재검색·라벨 → 공백 군집 → ID 없는 집계 갱신.

산출물 두 가지의 보관 기한이 다르다.
- `<snapshot>/analysis.json`: 군집별 관측 피처(정규화 좌표)까지 담는다. 스냅샷과 함께
  TTL로 지워진다.
- `<root>/aggregates/clusters.json`: 설치 k곳 이상 군집의 ID 없는 집계. 오래 남는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import aggregates as agg
from .cluster import cluster_observations
from .config import GapConfig
from .features import mirror
from .libraries import Library
from .observations import load_jsonl
from .replay import check_fidelity, replay, summarize
from .ttl import OBSERVATIONS

ANALYSIS = "analysis.json"
MEMBERS_IN_REPORT = 12


def run(snapshot: Path, production: Library, cfg: GapConfig, root: Path,
        curated: Library | None = None) -> dict:
    observations = load_jsonl(snapshot / OBSERVATIONS)
    fidelity = check_fidelity(observations, production, cfg)
    items = replay(observations, production, cfg, curated)
    gaps = [i for i in items if i.label == "gap_open"]

    clusters = []
    if gaps:
        clusters = cluster_observations(
            np.stack([g.feature for g in gaps]), np.stack([g.mask for g in gaps]),
            [g.obs.inst for g in gaps], [str(g.obs.coverage_class) for g in gaps],
            eps=cfg.eps_cluster, min_common=cfg.min_common_joints)
    records = agg.build(clusters, gaps, cfg, production.version)
    merged = agg.merge(agg.load(root), records, cfg.eps_cluster)
    agg.assert_id_free(merged)
    agg.save(root, merged)

    # 이전 주기의 같은 자리 군집과 합쳐졌으면 그 키를 보여 준다(키가 주기마다 흔들리지 않게).
    targets = [c for c in clusters if c.installations >= cfg.k_min_installations]
    keys = {id(c): (agg.match(merged, r, cfg.eps_cluster) or r)["cluster_key"]
            for c, r in zip(targets, records)}
    detail = []
    for cluster in clusters:
        members = [gaps[i] for i in cluster.members]
        shown = []
        for member, flip in list(zip(members, cluster.mirrored))[:MEMBERS_IN_REPORT]:
            feature, mask = mirror(member.feature, member.mask) if flip else (member.feature, member.mask)
            shown.append({"feature": np.round(feature, 4).tolist(), "mask": mask.tolist(),
                          "prod_pose": member.prod_pose,
                          "prod_distance": round(member.prod_distance, 4),
                          "action": member.obs.tags.get("action"),
                          "view": member.obs.tags.get("view")})
        detail.append({
            "cluster_key": keys.get(id(cluster)),
            "target": cluster.installations >= cfg.k_min_installations,
            "coverage_class": cluster.coverage_class,
            "installations": cluster.installations,
            "observations": len(members),
            "max_pair_distance": round(cluster.max_pair_distance, 4),
            "members": shown,
        })
    analysis = {
        "schema_version": 1,
        "thresholds_version": cfg.version,
        "production_library": production.version,
        "curated_library": curated.version if curated else None,
        "fidelity": fidelity,
        "summary": {**summarize(items), "clusters": len(clusters),
                    "target_clusters": sum(c["target"] for c in detail),
                    "aggregates_total": len(merged)},
        "clusters": detail,
    }
    (snapshot / ANALYSIS).write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    return analysis
