from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def hash_jsonl(rows: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str | Path, value) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    atomic_write_text(
        path,
        "".join(canonical_json(row) + "\n" for row in rows),
    )


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def resolve_path(value: str | Path, base: str | Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def relative_to_repo(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def slug(value: str) -> str:
    out = []
    dash = False
    for char in value.lower():
        if char.isalnum():
            out.append(char)
            dash = False
        elif not dash:
            out.append("-")
            dash = True
    result = "".join(out).strip("-")
    return result or "item"


def image_dimensions(path: str | Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def tree_fingerprint(path: str | Path) -> dict | None:
    root = Path(path)
    if not root.exists():
        return None
    if root.is_file():
        return {
            "path": relative_to_repo(root),
            "sha256": sha256_file(root),
            "files": 1,
            "bytes": root.stat().st_size,
        }
    digest = hashlib.sha256()
    count = 0
    total = 0
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = item.relative_to(root).as_posix()
        item_hash = sha256_file(item)
        size = item.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\0")
        count += 1
        total += size
    return {
        "path": relative_to_repo(root),
        "sha256": digest.hexdigest(),
        "files": count,
        "bytes": total,
    }


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args], cwd=root, stderr=subprocess.DEVNULL
    )


def git_snapshot(root: str | Path = REPO_ROOT) -> dict:
    repo = Path(root).resolve()
    try:
        sha = _git(repo, "rev-parse", "HEAD").decode().strip()
        status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
        diff = _git(repo, "diff", "--binary", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "dirty": None, "dirty_diff_sha256": None}

    dirty = bool(status.strip())
    fingerprint = hashlib.sha256()
    fingerprint.update(status)
    fingerprint.update(diff)
    if dirty:
        for raw_line in status.decode("utf-8", errors="replace").splitlines():
            if not raw_line.startswith("?? "):
                continue
            candidate = repo / raw_line[3:]
            if candidate.is_file():
                fingerprint.update(raw_line[3:].encode("utf-8"))
                fingerprint.update(sha256_file(candidate).encode("ascii"))
    return {
        "git_sha": sha,
        "dirty": dirty,
        "dirty_diff_sha256": fingerprint.hexdigest() if dirty else None,
    }


def runtime_snapshot() -> dict:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "os": platform.platform(),
        "machine": platform.machine(),
    }
