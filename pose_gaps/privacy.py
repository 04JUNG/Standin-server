"""로컬 파일에 사용자 식별자가 남아 있는지 찾는다.

    python -m pose_gaps privacy-scan data/curation config --strict

찾는 것: 사용자 입력 해시 접두사(`user_<12hex>` — `jobs.input_sha256`으로 작업까지
다시 연결된다), BFF 설치·작업 ID(`inst_<uuid>`, `job_<uuid>`), BetaData key 경로
(`installations/`), BetaData 버킷 이름. `--hash-list`로 사용자 입력 SHA-256 목록을 주면
그 값과 앞 12자도 찾는다. 목록은 메모리에만 두고 결과에도 값을 쓰지 않는다.

파일 이름·경로와, 텍스트·SQLite 같은 바이너리 내용까지 바이트로 훑는다. 이미지 내용은
건너뛴다(이름은 본다). 찾은 값 자체는 출력하지 않고 파일·종류·개수만 알린다.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Iterable

PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("user_input_hash", re.compile(rb"user_[0-9a-f]{12}")),
    ("installation_id", re.compile(rb"inst_[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE)),
    ("job_id", re.compile(rb"job_[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE)),
    ("beta_data_key", re.compile(rb"installations/")),
    ("beta_data_bucket", re.compile(rb"betadatabucket", re.IGNORECASE)),
)
SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", ".pytest_cache"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})
MAX_BYTES = 256 * 1024 * 1024
# 패턴 문자열이 들어 있는 이 패키지 자신은 기본으로 건너뛴다.
SELF_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Finding:
    path: str
    label: str
    count: int
    where: str          # name | content


def _excluded(path: Path, exclude: tuple[str, ...]) -> bool:
    resolved = path.resolve()
    if resolved == SELF_DIR or SELF_DIR in resolved.parents:
        return True
    posix = path.as_posix()
    return any(Path(posix).match(pattern) or pattern in posix for pattern in exclude)


def _files(paths: Iterable[Path], exclude: tuple[str, ...] = ()):
    for root in paths:
        root = Path(root)
        if root.is_file():
            if not _excluded(root, exclude):
                yield root
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                       and not _excluded(Path(directory) / d, exclude)]
            for name in files:
                path = Path(directory) / name
                if not _excluded(path, exclude):
                    yield path


def load_hash_list(path: Path) -> frozenset[bytes]:
    """SHA-256 목록(한 줄에 하나, 또는 JSON 배열/객체의 값)을 메모리에만 올린다."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
        values = payload if isinstance(payload, list) else list(_strings(payload))
    except ValueError:
        values = text.split()
    hashes = {str(v).lower() for v in values if re.fullmatch(r"[0-9a-fA-F]{64}", str(v))}
    return frozenset(h.encode() for h in hashes) | frozenset(h[:12].encode() for h in hashes)


def _strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def scan(paths: Iterable[Path], hashes: frozenset[bytes] = frozenset(),
         exclude: tuple[str, ...] = ()) -> list[Finding]:
    findings: list[Finding] = []
    for path in _files(paths, tuple(exclude)):
        name = str(path).replace("\\", "/").encode("utf-8", "replace")
        for label, pattern in PATTERNS:
            hits = len(pattern.findall(name))
            if hits:
                findings.append(Finding(str(path), label, hits, "name"))
        if path.suffix.lower() in IMAGE_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                findings.append(Finding(str(path), "too_large_to_scan", 1, "content"))
                continue
            data = path.read_bytes()
        except OSError:
            findings.append(Finding(str(path), "unreadable", 1, "content"))
            continue
        for label, pattern in PATTERNS:
            hits = len(pattern.findall(data))
            if hits:
                findings.append(Finding(str(path), label, hits, "content"))
        if hashes:
            hits = sum(data.count(value) for value in hashes)
            if hits:
                findings.append(Finding(str(path), "listed_user_hash", hits, "content"))
    return findings


def format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "privacy-scan: 0 findings"
    lines = [f"privacy-scan: {len(findings)} findings"]
    for f in findings:
        lines.append(f"  {f.label:<20} {f.where:<7} x{f.count:<5} {f.path}")
    return "\n".join(lines)
