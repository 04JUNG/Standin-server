"""라이브러리 DB를 읽기 전용으로 열고, 운영과 같은 경로로 검색한다.

`src.repo.load_entries`는 쓰기 모드로 연결해 스키마를 만들어 버리므로, 다른 세션의
정리 DB나 배포 번들을 건드리지 않도록 여기서는 `mode=ro`로 연다. 행 해석과 피처
검증은 `src.repo`의 것을 그대로 쓴다. 검색은 `src.search.knn_geometric`(운영 경로,
quarantine 포함)을 호출한다.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

import numpy as np

from src.repo import FEATURE_VERSION, _validated_feature
from src.schema import LibraryEntry, PoseCandidate, View
from src.search import PositionSearchIndex, knn_geometric, pose_family_id

MANIFEST_NAME = "library_manifest.json"


def load_entries_readonly(db_path: Path) -> list[LibraryEntry]:
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT p.pose_id, p.bvh_path, p.shot, p.action, p.relationship, p.meta_json,"
            "       pr.view, pr.feature_blob, pr.feature_version "
            "FROM pose_projections pr JOIN poses p ON p.pose_id = pr.pose_id"
        ).fetchall()
    finally:
        con.close()
    out = []
    for row in rows:
        if row["feature_version"] != FEATURE_VERSION:
            raise RuntimeError(
                f"feature_version 불일치(db={row['feature_version']} != code={FEATURE_VERSION})")
        feature = _validated_feature(
            np.frombuffer(row["feature_blob"], dtype=np.float32).copy(), row["pose_id"])
        tags = {"shot": row["shot"], "action": row["action"],
                "relationship": row["relationship"], "view": row["view"]}
        out.append(LibraryEntry(pose_id=row["pose_id"], view=View(row["view"]), feature=feature,
                                tags=tags, bvh_path=row["bvh_path"],
                                meta=json.loads(row["meta_json"] or "{}")))
    return out


def _version_from(path: Path) -> str | None:
    """번들 폴더면 library_manifest.json의 버전을 읽는다(내용 검증은 배포 검증기 몫)."""
    manifest = (path if path.is_dir() else path.parent) / MANIFEST_NAME
    if manifest.is_file():
        try:
            return str(json.loads(manifest.read_text(encoding="utf-8"))["library_version"])
        except (ValueError, KeyError):
            return None
    return None


@dataclass
class Library:
    path: Path
    version: str | None
    entries: list[LibraryEntry]
    index: PositionSearchIndex

    @classmethod
    def load(cls, path: Path | str, version: str | None = None) -> "Library":
        """`path`는 poses.db 파일이거나 그것을 담은 번들 폴더."""
        path = Path(path)
        db = path / "poses.db" if path.is_dir() else path
        if not db.is_file():
            raise FileNotFoundError(f"poses.db가 없습니다: {db}")
        entries = load_entries_readonly(db)
        return cls(path=path, version=version or _version_from(path), entries=entries,
                   index=PositionSearchIndex.build(entries))

    @property
    def pose_ids(self) -> set[str]:
        return {entry.pose_id for entry in self.entries}

    @property
    def family_by_pose(self) -> dict[str, str]:
        return {entry.pose_id: pose_family_id(entry.pose_id, entry.meta) for entry in self.entries}

    def search(self, feature: np.ndarray, mask: np.ndarray, *, metric: str = "pos",
               top_k: int = 5) -> list[PoseCandidate]:
        return knn_geometric(self.entries, np.asarray(feature, dtype=np.float32).reshape(-1),
                             top_k=top_k, query_valid_mask=mask,
                             search_index=self.index, metric=metric)

    def top1(self, feature, mask, *, metric: str = "pos") -> PoseCandidate | None:
        hits = self.search(feature, mask, metric=metric, top_k=1)
        return hits[0] if hits else None
