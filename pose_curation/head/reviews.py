"""Local face-angle review history, isolated from pose-library publication."""

import json
import sqlite3
from contextlib import contextmanager

from ..storage import utc_now


class HeadReviewStore:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                """CREATE TABLE IF NOT EXISTS head_angle_events (
                revision INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT NOT NULL, target TEXT NOT NULL, scope TEXT NOT NULL,
                record TEXT NOT NULL)"""
            )
            con.execute(
                """CREATE INDEX IF NOT EXISTS head_angle_lookup
                ON head_angle_events(image_hash, target, scope, revision)"""
            )

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=10)
        try:
            with con:
                yield con
        finally:
            con.close()

    def list(self, image_hash):
        with self.connection() as con:
            rows = con.execute(
                """SELECT revision, record FROM head_angle_events
                WHERE revision IN (SELECT MAX(revision) FROM head_angle_events
                    WHERE image_hash=? GROUP BY target, scope)
                ORDER BY revision DESC""",
                (image_hash,),
            ).fetchall()
        return [
            {**json.loads(record), "revision": revision} for revision, record in rows
        ]

    def save(self, record, expected_revision):
        identity = (record["image_sha256"], record["target"], record["scope"])
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            current = con.execute(
                """SELECT COALESCE(MAX(revision), 0)
                FROM head_angle_events WHERE image_hash=? AND target=? AND scope=?""",
                identity,
            ).fetchone()[0]
            if current != expected_revision:
                raise ValueError(
                    "다른 검수 기록이 저장되었습니다. 목록을 다시 불러와 주세요."
                )
            record = {**record, "updated_at": utc_now()}
            cursor = con.execute(
                """INSERT INTO head_angle_events
                (image_hash, target, scope, record) VALUES (?,?,?,?)""",
                (*identity, json.dumps(record, ensure_ascii=False)),
            )
            return {**record, "revision": cursor.lastrowid}
