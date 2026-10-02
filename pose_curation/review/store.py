"""Durable review decisions, bound to the exact BVH content being reviewed."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import json

from ..storage import utc_now


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
                CREATE TABLE IF NOT EXISTS reviews (
                    pose_key TEXT NOT NULL, content_hash TEXT NOT NULL,
                    status TEXT NOT NULL, note TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (pose_key, content_hash)
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    id INTEGER PRIMARY KEY, pose_key TEXT NOT NULL, content_hash TEXT NOT NULL,
                    status TEXT NOT NULL, note TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS visual_evidence (
                    pose_key TEXT NOT NULL, content_hash TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    PRIMARY KEY (pose_key, content_hash)
                );
                CREATE TABLE IF NOT EXISTS visual_evidence_events (
                    id INTEGER PRIMARY KEY, pose_key TEXT NOT NULL,
                    content_hash TEXT NOT NULL, updated_at TEXT NOT NULL,
                    evidence_json TEXT NOT NULL
                );
            """)

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def all(self) -> dict[tuple[str, str], dict]:
        with self.connection() as con:
            rows = {
                (row["pose_key"], row["content_hash"]): dict(row)
                for row in con.execute("SELECT * FROM reviews")
            }
            for row in con.execute("SELECT * FROM visual_evidence"):
                key = (row["pose_key"], row["content_hash"])
                if key in rows:
                    rows[key]["evidence"] = json.loads(row["evidence_json"])
            return rows

    def save(
        self, key: str, content_hash: str, status: str, note: str, *, evidence=None
    ) -> dict:
        record = {
            "pose_key": key,
            "content_hash": content_hash,
            "status": status,
            "note": note,
            "updated_at": utc_now(),
        }
        with self.connection() as con:
            con.execute(
                "INSERT INTO reviews VALUES (:pose_key,:content_hash,:status,:note,:updated_at) "
                "ON CONFLICT(pose_key,content_hash) DO UPDATE SET status=excluded.status, "
                "note=excluded.note,updated_at=excluded.updated_at",
                record,
            )
            con.execute(
                "INSERT INTO review_events (pose_key,content_hash,status,note,updated_at) "
                "VALUES (:pose_key,:content_hash,:status,:note,:updated_at)",
                record,
            )
            if evidence is not None:
                con.execute(
                    "INSERT INTO visual_evidence_events(pose_key,content_hash,updated_at,evidence_json) VALUES (?,?,?,?)",
                    (
                        key,
                        content_hash,
                        record["updated_at"],
                        json.dumps(evidence, ensure_ascii=False),
                    ),
                )
                con.execute(
                    "INSERT INTO visual_evidence VALUES (?,?,?) ON CONFLICT(pose_key,content_hash) DO UPDATE SET evidence_json=excluded.evidence_json",
                    (key, content_hash, json.dumps(evidence, ensure_ascii=False)),
                )
                record["evidence"] = evidence
        return record
