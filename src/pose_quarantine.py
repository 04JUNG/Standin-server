"""Release-time pose quarantine shared by geometry search and API delivery."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .config import CFG


@lru_cache(maxsize=16)
def _load_cached(resolved_path: str, mtime_ns: int) -> dict[str, dict]:
    del mtime_ns  # cache key에 파일 변경 시각을 포함하기 위한 값이다.
    path = Path(resolved_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid pose quarantine: {path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported pose quarantine schema: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError(f"pose quarantine entries must be a list: {path}")
    records: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError(f"pose quarantine entry must be an object: {path}")
        pose_id = entry.get("pose_id")
        reason = entry.get("reason")
        if not isinstance(pose_id, str) or not pose_id.strip():
            raise RuntimeError(f"pose quarantine pose_id is required: {path}")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError(f"pose quarantine reason is required: {pose_id}")
        if pose_id in records:
            raise RuntimeError(f"duplicate pose quarantine entry: {pose_id}")
        records[pose_id] = dict(entry)
    return records


def load_pose_quarantine(cfg=CFG) -> dict[str, dict]:
    raw = str(getattr(cfg, "refine_pose_quarantine_path", "")).strip()
    if not raw:
        return {}
    path = Path(raw).expanduser().resolve()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError as exc:
        raise RuntimeError(f"pose quarantine file unavailable: {path}: {exc}") from exc
    return _load_cached(str(path), mtime_ns)


def quarantine_record(pose_id: str, cfg=CFG) -> dict | None:
    return load_pose_quarantine(cfg).get(str(pose_id))


def is_pose_quarantined(pose_id: str, cfg=CFG) -> bool:
    return quarantine_record(pose_id, cfg) is not None


def pose_quarantine_sha256(cfg=CFG) -> str:
    raw = str(getattr(cfg, "refine_pose_quarantine_path", "")).strip()
    if not raw:
        return hashlib.sha256(b"").hexdigest()
    path = Path(raw).expanduser().resolve()
    # 먼저 schema를 검증해 잘못된 정책 파일이 정상 fingerprint처럼 보이지 않게 한다.
    load_pose_quarantine(cfg)
    return hashlib.sha256(path.read_bytes()).hexdigest()
