"""
포즈 라이브러리 저장소(SQLite).

역할 분리:
  - DB = 라이브러리의 '단일 소스'(태그·피처·BVH 경로·라이선스). 영속성과 공유 아티팩트.
  - kNN 자체는 MVP 규모(수천 벡터)에선 메모리 브루트포스로 충분 → search.py 그대로 재사용.
    (스케일이 커지면 pgvector/LanceDB로 이 레이어만 교체. 검색 인터페이스는 불변.)

스키마:
  poses(pose_id PK, bvh_path, source, license, shot, action, relationship,
        set_id, set_role, meta_json)
  pose_projections(id PK, pose_id FK, view, feature_blob, feature_version)

feature는 np.float32 배열을 bytes로 저장(BLOB). 복원 시 np.frombuffer.
feature_version: 정규화 방식이 바뀌면 올린다 — 쿼리/라이브러리 정규화 불일치를 잡는 안전장치.
"""
from __future__ import annotations

import json
import sqlite3
from typing import List, Optional

import numpy as np

from .schema import LibraryEntry, View

FEATURE_VERSION = 1   # features.normalize_skeleton 규격 버전. 바꾸면 +1 하고 재빌드.
FEATURE_SHAPE = (34,)


def _validated_feature(feature, pose_id: str) -> np.ndarray:
    feat = np.asarray(feature, dtype=np.float32).reshape(-1)
    if feat.shape != FEATURE_SHAPE:
        raise ValueError(
            f"pose '{pose_id}' feature shape must be {FEATURE_SHAPE}, got {feat.shape}"
        )
    if not np.isfinite(feat).all():
        raise ValueError(f"pose '{pose_id}' feature contains NaN/Inf")
    return feat

_SCHEMA = """
CREATE TABLE IF NOT EXISTS poses (
    pose_id      TEXT PRIMARY KEY,
    bvh_path     TEXT,
    source       TEXT,          -- cmu | mixamo | custom
    license      TEXT,          -- 재배포 조항 메모(CMU/Mixamo 주의)
    shot         TEXT,
    action       TEXT,
    relationship TEXT,
    set_id       TEXT,          -- 얽힘 상호작용 그룹(nullable). 같은 set_id=한 장면
    set_role     TEXT,          -- 세트 내 역할(예: A/B). solo면 NULL
    meta_json    TEXT
);
CREATE TABLE IF NOT EXISTS pose_projections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pose_id         TEXT NOT NULL REFERENCES poses(pose_id),
    view            TEXT NOT NULL,
    feature_blob    BLOB NOT NULL,
    feature_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proj_pose ON pose_projections(pose_id);
CREATE INDEX IF NOT EXISTS idx_poses_tags ON poses(action, relationship, shot);
"""


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    return con


def build_db(entries: List[LibraryEntry], db_path: str) -> int:
    """LibraryEntry 목록 → SQLite. 같은 pose_id는 poses 1행 + view별 projections N행."""
    # 기존 DB를 비우기 전에 전 항목을 검증한다. 중간 feature 하나가 깨졌다고 정상 DB를
    # 부분 갱신하거나 열린 transaction으로 남기면 안 된다.
    validated = [(entry, _validated_feature(entry.feature, entry.pose_id))
                 for entry in entries]
    con = connect(db_path)
    con.execute("DELETE FROM pose_projections")
    con.execute("DELETE FROM poses")
    seen = set()
    for e, validated_feature in validated:
        if e.pose_id not in seen:
            con.execute(
                "INSERT INTO poses(pose_id,bvh_path,source,license,shot,action,"
                "relationship,set_id,set_role,meta_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (e.pose_id, e.bvh_path, e.meta.get("source", "synthetic"),
                 e.meta.get("license", "n/a"), e.tags.get("shot"),
                 e.tags.get("action"), e.tags.get("relationship"),
                 e.meta.get("set_id"), e.meta.get("set_role"),
                 json.dumps(e.meta, ensure_ascii=False)),
            )
            seen.add(e.pose_id)
        feat = validated_feature.tobytes()
        con.execute(
            "INSERT INTO pose_projections(pose_id,view,feature_blob,feature_version)"
            " VALUES(?,?,?,?)",
            (e.pose_id, e.view.value, feat, FEATURE_VERSION),
        )
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM pose_projections").fetchone()[0]
    con.close()
    return n


def load_entries(db_path: str) -> List[LibraryEntry]:
    """DB → LibraryEntry 목록(메모리 적재). search.knn이 그대로 소비."""
    con = connect(db_path)
    rows = con.execute(
        "SELECT p.pose_id, p.bvh_path, p.shot, p.action, p.relationship, p.meta_json,"
        "       pr.view, pr.feature_blob, pr.feature_version "
        "FROM pose_projections pr JOIN poses p ON p.pose_id = pr.pose_id"
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        if r["feature_version"] != FEATURE_VERSION:
            # 정규화 규격 불일치 → 재빌드 필요. 조용히 쓰지 말고 알린다.
            raise RuntimeError(
                f"feature_version 불일치(db={r['feature_version']} != "
                f"code={FEATURE_VERSION}). DB를 재빌드하세요(scripts/build_db.py).")
        feat = _validated_feature(
            np.frombuffer(r["feature_blob"], dtype=np.float32).copy(),
            r["pose_id"],
        )
        tags = {"shot": r["shot"], "action": r["action"],
                "relationship": r["relationship"], "view": r["view"]}
        out.append(LibraryEntry(
            pose_id=r["pose_id"], view=View(r["view"]), feature=feat,
            tags=tags, bvh_path=r["bvh_path"],
            meta=json.loads(r["meta_json"] or "{}"),
        ))
    return out


def get_bvh_path(db_path: str, pose_id: str) -> Optional[str]:
    """동원 핸드오프: pose_id → 라이브러리 BVH 파일 경로."""
    con = connect(db_path)
    row = con.execute("SELECT bvh_path FROM poses WHERE pose_id=?", (pose_id,)).fetchone()
    con.close()
    return row["bvh_path"] if row else None


def get_pose_meta(db_path: str, pose_id: str) -> Optional[dict]:
    """export order 강화용: pose_id → {shot, action, relationship, set_id, set_role, bvh_path}. (BVH는 1인)"""
    con = connect(db_path)
    row = con.execute(
        "SELECT shot, action, relationship, set_id, set_role, bvh_path "
        "FROM poses WHERE pose_id=?", (pose_id,)).fetchone()
    con.close()
    if not row:
        return None
    return {"shot": row["shot"], "action": row["action"],
            "relationship": row["relationship"], "set_id": row["set_id"],
            "set_role": row["set_role"], "bvh_path": row["bvh_path"]}
