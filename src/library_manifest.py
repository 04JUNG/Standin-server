"""포즈 라이브러리 번들의 내용 주소 manifest.

S3 key는 고정(`pose-library/v1.tar.gz`)이고 `POSE_LIBRARY_VERSION`은 인프라에 "v1"로
박혀 있어서, 어느 번들이 검색에 답했는지 응답만 보고는 알 수 없다. 그래서 번들 루트에
`library_manifest.json`을 두고 버전을 **내용 해시에서** 만든다. 같은 내용이면 같은 버전,
한 바이트라도 다르면 다른 버전이다.

- 만들기: `scripts/build_pose_bundle.py`가 쓴다.
- 확인: `scripts/deploy_pose_library.py`가 업로드 전에 해시를 다시 계산한다.
- 기동: 서버가 manifest를 읽어 응답의 `pose_library_version`에 싣는다(후속 변경).
  manifest가 없는 옛 번들로 롤백해도 기동은 되어야 하므로 env 값으로 폴백한다.

내용 해시의 대상은 서버가 실제로 쓰는 파일뿐이다: `poses.db`, `bvh/**`, `thumbs/**`,
`ATTRIBUTION.md`. manifest 자신과 `index.pkl`(서버가 읽지 않는 레거시)은 넣지 않는다.

`semantic/`(04J-285 자연어 검색 태그)은 **내용 해시에서 뺀다.** 태그만 고쳐도
`library_version`이 바뀌면 기하 검색 재생 게이트를 다시 돌려야 하는데, 태그는 기하 검색에
영향이 없다. 대신 manifest의 `semantic` 절에 파일 해시와 기준 `db_sha256`을 따로 남기고
배포 검증기가 같은 방식으로 다시 잰다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from .repo import FEATURE_VERSION

MANIFEST_NAME = "library_manifest.json"
SCHEMA_VERSION = 1
BUNDLE_DB = "poses.db"
BUNDLE_DIRS = ("bvh", "thumbs")
BUNDLE_FILES = ("ATTRIBUTION.md",)
SKIP_NAMES = frozenset({"__pycache__", ".DS_Store", "Thumbs.db"})
VERSION_PATTERN = re.compile(r"^lib-(\d{8})-([0-9a-f]{8})$")
LEGACY_VERSION = "v1"

# replay_gate.status 중 배포를 막지 않는 값. not_required는 부모와 내용이 같은 번들
# (예: manifest만 새로 붙인 기준 번들)에만 쓴다.
GATE_OK = frozenset({"passed", "not_required"})

SEMANTIC_DIR = "semantic"
SEMANTIC_TAGS = "search-v2.json"
SEMANTIC_TAXONOMY = "search-taxonomy-v2.json"
SEMANTIC_FILES = (SEMANTIC_TAGS, SEMANTIC_TAXONOMY)


class LibraryManifestError(RuntimeError):
    """manifest가 없거나, 형식이 틀리거나, 번들 내용과 맞지 않는다."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk(directory: Path, prefix: str):
    for child in sorted(directory.iterdir(), key=lambda p: p.name):
        if child.name in SKIP_NAMES:
            continue
        rel = f"{prefix}/{child.name}"
        if child.is_dir():
            yield from _walk(child, rel)
        elif child.is_file():
            yield rel, child


def bundle_files(bundle_dir: Path) -> list[tuple[str, Path]]:
    """내용 해시 대상 파일을 (posix 상대경로, 실제 경로)로 정렬해 돌려준다."""
    root = Path(bundle_dir)
    db = root / BUNDLE_DB
    if not db.is_file():
        raise LibraryManifestError(f"{db} 가 없습니다")
    files = [(BUNDLE_DB, db)]
    for name in BUNDLE_DIRS:
        if (root / name).is_dir():
            files.extend(_walk(root / name, name))
    for name in BUNDLE_FILES:
        if (root / name).is_file():
            files.append((name, root / name))
    return sorted(files, key=lambda item: item[0])


def content_sha256(bundle_dir: Path) -> str:
    """`<relpath>\\t<sha256>\\n` 줄을 경로순으로 이어 붙인 것의 SHA-256."""
    digest = hashlib.sha256()
    for rel, path in bundle_files(bundle_dir):
        digest.update(f"{rel}\t{sha256_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def library_version(content_sha: str, created_at: datetime) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", content_sha or ""):
        raise LibraryManifestError("content_sha256 must be 64 lowercase hex characters")
    return f"lib-{created_at.astimezone(timezone.utc):%Y%m%d}-{content_sha[:8]}"


def count_bundle(bundle_dir: Path) -> dict[str, Any]:
    root = Path(bundle_dir)
    con = sqlite3.connect((root / BUNDLE_DB).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        poses = con.execute("SELECT COUNT(*) FROM poses").fetchone()[0]
        projections = con.execute("SELECT COUNT(*) FROM pose_projections").fetchone()[0]
        by_source = dict(con.execute(
            "SELECT COALESCE(source, 'unknown'), COUNT(*) FROM poses GROUP BY 1 ORDER BY 1"))
        groups: dict[str, int] = {}
        for (raw,) in con.execute("SELECT meta_json FROM poses"):
            try:
                group = json.loads(raw or "{}").get("curation_group") or "unlabelled"
            except ValueError:
                group = "unlabelled"
            groups[group] = groups.get(group, 0) + 1
    finally:
        con.close()
    bvh = sum(1 for p in (root / "bvh").glob("*.bvh")) if (root / "bvh").is_dir() else 0
    thumbs = sum(1 for p in (root / "thumbs").glob("*.jpg")) if (root / "thumbs").is_dir() else 0
    return {"poses": poses, "projections": projections, "bvh": bvh, "thumbs": thumbs,
            "by_source": by_source, "by_group": dict(sorted(groups.items()))}


def build_manifest(bundle_dir: Path, *, created_at: datetime | None = None,
                   parent: dict | None = None, curation: dict | None = None,
                   replay_gate: dict | None = None, privacy_scan: dict | None = None,
                   compat: dict | None = None, content: str | None = None) -> dict[str, Any]:
    """번들 폴더의 현재 내용으로 manifest dict를 만든다(파일은 쓰지 않는다).

    `content`는 호출자가 방금 같은 폴더로 계산한 `content_sha256` 값이다. 파일 수천 개를
    다시 해시하지 않으려고 받는다. 모르면 비워 두면 여기서 계산한다.
    """
    root = Path(bundle_dir)
    created = created_at or datetime.now(timezone.utc)
    content = content or content_sha256(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "library_version": library_version(content, created),
        "created_at": created.astimezone(timezone.utc).isoformat(),
        "content_sha256": content,
        "db_sha256": sha256_file(root / BUNDLE_DB),
        "feature_version": FEATURE_VERSION,
        "counts": count_bundle(root),
        "parent": parent or {"library_version": LEGACY_VERSION, "content_sha256": None},
        "curation": curation or {},
        "replay_gate": replay_gate or {"status": "not_run"},
        "privacy_scan": privacy_scan or {"status": "not_run"},
        "compat": compat or {"status": "not_run"},
    }


def write_manifest(bundle_dir: Path, manifest: dict) -> Path:
    path = Path(bundle_dir) / MANIFEST_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_manifest(bundle_dir: Path) -> dict | None:
    path = Path(bundle_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise LibraryManifestError(f"{path} 를 JSON으로 읽지 못했습니다: {exc}") from exc
    if not isinstance(payload, dict):
        raise LibraryManifestError(f"{path} 는 JSON 객체여야 합니다")
    return payload


def verify_manifest(bundle_dir: Path, manifest: dict | None = None, *,
                    check_content: bool = True, check_semantic: bool = True) -> dict:
    """manifest 형식과 번들 내용의 일치를 확인한다. 문제가 있으면 전부 모아 한 번에 알린다."""
    root = Path(bundle_dir)
    payload = manifest if manifest is not None else read_manifest(root)
    if payload is None:
        raise LibraryManifestError(f"{root / MANIFEST_NAME} 가 없습니다")

    problems: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {payload.get('schema_version')!r} != {SCHEMA_VERSION}")
    version = str(payload.get("library_version") or "")
    match = VERSION_PATTERN.fullmatch(version)
    content = str(payload.get("content_sha256") or "")
    if not match:
        problems.append(f"library_version 형식이 틀립니다: {version!r} (lib-YYYYMMDD-<hash8>)")
    elif content[:8] != match.group(2):
        problems.append("library_version의 해시 부분이 content_sha256과 다릅니다")
    if payload.get("feature_version") != FEATURE_VERSION:
        problems.append(
            f"feature_version {payload.get('feature_version')!r} != src/repo.py {FEATURE_VERSION}")
    db = root / BUNDLE_DB
    if not db.is_file():
        problems.append(f"{db} 가 없습니다")
    elif payload.get("db_sha256") != sha256_file(db):
        problems.append("db_sha256이 poses.db와 다릅니다(manifest를 만든 뒤 DB가 바뀌었습니다)")
    if check_semantic:
        problems.extend(verify_semantic_section(root, payload))
    if check_content and not problems:
        actual = content_sha256(root)
        if actual != content:
            problems.append("content_sha256이 번들 내용과 다릅니다(bvh/·thumbs/가 바뀌었습니다)")
    if problems:
        raise LibraryManifestError("library manifest 검증 실패:\n  - " + "\n  - ".join(problems))
    return payload


def semantic_files_sha256(bundle_dir: Path) -> dict[str, str] | None:
    """`semantic/` 파일 해시. 폴더가 없으면 None, 일부만 있으면 오류."""
    root = Path(bundle_dir) / SEMANTIC_DIR
    if not root.exists():
        return None
    missing = [name for name in SEMANTIC_FILES if not (root / name).is_file()]
    if missing:
        raise LibraryManifestError(f"{root} 에 {', '.join(missing)} 가 없습니다")
    return {name: sha256_file(root / name) for name in SEMANTIC_FILES}


def verify_semantic_section(bundle_dir: Path, manifest: dict) -> list[str]:
    """manifest.semantic과 실제 semantic/ 파일·poses.db가 맞는지. 문제 목록을 돌려준다."""
    root = Path(bundle_dir)
    section = manifest.get("semantic")
    try:
        actual = semantic_files_sha256(root)
    except LibraryManifestError as exc:
        return [str(exc)]
    if actual is None:
        return ["manifest에 semantic 절이 있는데 semantic/ 폴더가 없습니다"] if section else []
    if not section:
        return ["semantic/ 폴더가 있는데 manifest에 기록되지 않았습니다"
                "(build_pose_bundle.py attach-semantic으로 붙이세요)"]
    problems = []
    if section.get("files") != actual:
        problems.append("semantic/ 파일 해시가 manifest와 다릅니다")
    db = root / BUNDLE_DB
    if db.is_file() and section.get("source_db_sha256") != sha256_file(db):
        problems.append("semantic 태그가 다른 poses.db 기준으로 만들어졌습니다")
    if section.get("status") != "passed":
        problems.append(f"semantic.status={section.get('status')!r} (passed여야 함)")
    return problems


@dataclass(frozen=True)
class LibraryIdentity:
    version: str
    source: str            # "manifest" | "env"
    content_sha256: str | None = None
    db_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "source": self.source,
                "content_sha256": self.content_sha256, "db_sha256": self.db_sha256}


def resolve_library_identity(data_dir: Path, *, fallback_version: str,
                             strict: bool = False,
                             check_content: bool = False) -> LibraryIdentity:
    """기동 시 쓸 라이브러리 식별자.

    manifest가 없으면 env 값(`POSE_LIBRARY_VERSION`)으로 폴백한다. manifest는 있는데
    내용과 맞지 않으면 strict(운영)에서는 예외, 아니면 env로 폴백한다. 맞지 않는
    manifest의 버전을 그대로 내보내면 측정이 엉뚱한 번들에 귀속된다.
    """
    root = Path(data_dir)
    try:
        payload = read_manifest(root)
        if payload is None:
            return LibraryIdentity(fallback_version, "env")
        # 시맨틱 태그 불일치로 기하 검색 기동을 막지 않는다. 서버의 시맨틱 로더가 따로
        # db_sha256을 확인하고, 틀리면 /semantic-search만 503이 된다.
        verify_manifest(root, payload, check_content=check_content, check_semantic=False)
    except LibraryManifestError:
        if strict:
            raise
        return LibraryIdentity(fallback_version, "env")
    return LibraryIdentity(str(payload["library_version"]), "manifest",
                           str(payload["content_sha256"]), str(payload["db_sha256"]))


def gate_allows_deploy(manifest: dict) -> bool:
    return str((manifest.get("replay_gate") or {}).get("status")) in GATE_OK
