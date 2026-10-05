"""로컬 스냅샷 보관 기한.

개별 관측은 작업에 연결된 운영 데이터의 사본이다. 운영에서 작업이 지워지거나 동의가
철회되면 다음 pull부터 빠지지만, 이미 받은 스냅샷에는 남는다. 그래서 두 가지를 강제한다.

- 스냅샷은 `ttl_days`(기본 14일)가 지나면 지운다.
- 분석 명령은 `max_snapshot_age_days`(기본 7일)보다 오래된 스냅샷으로는 돌지 않는다.

둘을 합치면 운영 삭제가 로컬에 반영되기까지 최대 `ttl_days`가 걸린다. 항목별
`expires_on`(관측 + 365일)이 지난 행은 기한 안의 스냅샷에서도 지운다.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil

from .config import REPO_ROOT

SNAPSHOTS = "snapshots"
OBSERVATIONS = "observations.jsonl"
EXPORT_META = "export.json"


class StaleSnapshotError(RuntimeError):
    """쓸 수 있는(충분히 최근인) 스냅샷이 없다."""


def data_root(explicit: Path | str | None = None) -> Path:
    return Path(explicit or os.getenv("POSE_GAPS_DATA_DIR") or REPO_ROOT / "data" / "gaps")


def pulled_at(snapshot: Path) -> datetime:
    meta = json.loads((snapshot / EXPORT_META).read_text(encoding="utf-8"))
    return datetime.fromisoformat(meta["pulled_at"])


def list_snapshots(root: Path) -> list[tuple[datetime, Path]]:
    base = Path(root) / SNAPSHOTS
    if not base.is_dir():
        return []
    found = []
    for child in base.iterdir():
        if child.is_dir() and not child.name.startswith(".") and (child / EXPORT_META).is_file():
            found.append((pulled_at(child), child))
    return sorted(found)


def drop_expired_rows(snapshot: Path, today: date) -> int:
    """`expires_on`이 지난 행을 스냅샷 파일에서 지운다. 지운 행 수를 돌려준다."""
    path = snapshot / OBSERVATIONS
    if not path.is_file():
        return 0
    kept, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if date.fromisoformat(json.loads(line)["expires_on"]) <= today:
            dropped += 1
        else:
            kept.append(line)
    if dropped:
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    return dropped


def purge(root: Path, *, now: datetime, ttl_days: int) -> dict:
    """기한이 지난 스냅샷을 통째로, 남은 스냅샷에서는 만료 행을 지운다."""
    removed, dropped_rows = [], 0
    for taken, snapshot in list_snapshots(root):
        if now - taken > timedelta(days=ttl_days):
            shutil.rmtree(snapshot)
            removed.append(snapshot.name)
        else:
            dropped_rows += drop_expired_rows(snapshot, now.date())
    base = Path(root) / SNAPSHOTS
    if base.is_dir():  # 중단된 pull이 남긴 임시 폴더도 지운다
        for child in base.iterdir():
            if child.is_dir() and child.name.startswith(".tmp-"):
                shutil.rmtree(child)
    return {"removed_snapshots": removed, "dropped_rows": dropped_rows}


def latest_fresh(root: Path, *, now: datetime, max_age_days: int) -> Path:
    snapshots = list_snapshots(root)
    if not snapshots:
        raise StaleSnapshotError(f"{Path(root) / SNAPSHOTS} 에 스냅샷이 없습니다. 먼저 pull 하세요")
    taken, snapshot = snapshots[-1]
    if now - taken > timedelta(days=max_age_days):
        raise StaleSnapshotError(
            f"가장 최근 스냅샷이 {(now - taken).days}일 전입니다(한도 {max_age_days}일). "
            "운영에서 지워진 데이터가 남아 있을 수 있으니 다시 pull 하세요")
    return snapshot


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
