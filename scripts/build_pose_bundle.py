"""포즈 라이브러리 배포 번들을 만든다 — `library_manifest.json`까지 한 번에.

    # 지금 S3에 있는 번들을 내용 그대로 다시 묶고 manifest만 붙인다(검색 결과 불변).
    python scripts/build_pose_bundle.py baseline \
        --source data/_s3/pose-library-v1.tar.gz --out data/bundles/baseline

    # 로컬 정리 DB(`python -m pose_curation publish` 결과)를 배포 형식으로 바꾼다.
    python scripts/build_pose_bundle.py curated \
        --curated-db data/curation/library/poses.db --curation-dir data/curation \
        --base data/_s3/pose-library-v1.tar.gz --out data/bundles/next \
        --exclude-unresolved-rigs --exclude-private-identifiers

    # 재생 게이트 보고서를 manifest에 기록한다(배포 검증기가 이 결과를 본다).
    python scripts/build_pose_bundle.py record-gate \
        --bundle data/bundles/next --report <gate-report.json>

정리 DB를 그대로 올리면 안 되는 이유:
- `bvh_path`가 로컬 절대경로다. 컨테이너는 `data/bvh/<파일>.bvh`만 찾는다.
- meta에 로컬 경로·검수 참고 러프·보정 입력 해시 같은 내부 값이 섞여 있다.
- 신규 포즈 썸네일은 512px 캐릭터 렌더다. 서버 번들은 256px JPEG(q78)다.

이 스크립트는 `pose_curation`을 import하지 않는다. 정리 DB와 배치 manifest의 **파일
형식**만 읽는다(`pose_curation/publication.py`, `review/catalog.py`가 쓰는 형식). 마지막에
배포 검증기(`deploy_pose_library.validate`)를 그대로 돌려, 통과한 번들에만 manifest를 쓴다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

from scripts.deploy_pose_library import DeployError, validate
from scripts.pose_bundle_policy import converter_profile, privacy_findings, sanitize_meta
from src.library_manifest import (
    LEGACY_VERSION,
    SKIP_NAMES,
    LibraryManifestError,
    build_manifest,
    content_sha256,
    read_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)
from src.thumbnails import THUMBNAIL_VIEWS, thumbnail_filename

THUMB_SIZE = 256
THUMB_QUALITY = 78
NEW_THUMB_SOURCE = "curation_character_v1"
# 기존 운영 BVH에는 "Jumping Down (1)_00028.bvh"처럼 괄호가 들어간다.
# 경로 구분자는 여전히 거부하면서 배포 중인 파일 이름은 보존한다.
SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.()\-]*$")
VIEW_ORDER = ("front", "three_quarter", "side", "back")


class BundleError(RuntimeError):
    """번들을 만들지 못했다. 출력 폴더에는 manifest가 쓰이지 않는다."""


# ── 공통 ──────────────────────────────────────────────────────────────
def _prepare_out(out: Path, force: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not force:
            raise BundleError(f"{out} 가 비어 있지 않습니다. 덮어쓰려면 --force")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def _materialize(source: Path, workdir: Path) -> Path:
    """폴더면 그대로, `.tar.gz`면 workdir에 경로 탈출을 막으며 푼다."""
    if source.is_dir():
        return source
    if source.is_file() and source.name.endswith((".tar.gz", ".tgz")):
        from src.library_source import _safe_extract

        target = workdir / "source"
        target.mkdir()
        _safe_extract(source, target)
        return target
    raise BundleError(f"번들 폴더나 .tar.gz가 아닙니다: {source}")


def _parent_of(source_dir: Path, fallback_version: str) -> dict:
    """원본 번들의 식별자. manifest가 있으면 그 값을, 없으면 내용 해시를 직접 잰다."""
    manifest = read_manifest(source_dir)
    if manifest is not None:
        verify_manifest(source_dir, manifest)
        return {"library_version": manifest["library_version"],
                "content_sha256": manifest["content_sha256"]}
    return {"library_version": fallback_version, "content_sha256": content_sha256(source_dir)}


def _finish(out: Path, *, parent: dict, curation: dict, replay_gate: dict,
            compat: dict, content: str | None = None) -> dict:
    """배포 검증기를 manifest 없이 돌리고, 통과하면 manifest를 쓴다.

    `content`는 방금 계산한 out의 내용 해시다. 작은 파일 수천 개를 Windows에서 거듭
    해시하면 몇 분이 걸려서, 한 번 잰 값을 넘겨받는다. 배포 검증기는 업로드 전에 따로 다시 잰다.
    """
    try:
        summary = validate(out, allow_no_manifest=True)
    except DeployError as exc:
        raise BundleError(str(exc)) from exc
    manifest = build_manifest(
        out,
        parent=parent,
        curation=curation,
        replay_gate=replay_gate,
        privacy_scan={"status": "passed", "poses": summary["poses"],
                      "policy": "scripts/pose_bundle_policy.py"},
        compat=compat,
        content=content,
    )
    write_manifest(out, manifest)
    # 내용 해시는 build_manifest가 방금 같은 폴더로 정했다. 여기서는 형식·DB 해시만 본다.
    verify_manifest(out, manifest, check_content=False)
    return manifest


def _print_done(out: Path, manifest: dict) -> None:
    counts = manifest["counts"]
    print(f"\n번들 완성: {out}")
    print(f"  library_version  {manifest['library_version']}")
    print(f"  포즈 {counts['poses']}개 · 투영 {counts['projections']}개 · "
          f"썸네일 {counts['thumbs']}개")
    print(f"  replay_gate      {manifest['replay_gate'].get('status')}")
    print(f"  → 검증: python scripts/deploy_pose_library.py {out.as_posix()} --dry-run")


# ── baseline ──────────────────────────────────────────────────────────
_PAYLOAD = ("poses.db", "bvh", "thumbs", "ATTRIBUTION.md")


def build_baseline(source: Path, out: Path, *, parent_version: str = LEGACY_VERSION,
                   force: bool = False) -> dict:
    """원본 번들의 poses.db·bvh/·thumbs/·ATTRIBUTION.md를 바이트 그대로 옮기고 manifest를 붙인다.

    tar.gz는 출력 폴더에 바로 풀고 번들 밖 파일(index.pkl 등)만 지운다. 압축을 푼 바이트가
    곧 원본이므로 부모 내용 해시는 출력 폴더를 한 번 재서 얻는다.
    """
    _prepare_out(out, force)
    if source.is_file() and source.name.endswith((".tar.gz", ".tgz")):
        from src.library_source import _safe_extract

        _safe_extract(source, out)
        parent_manifest = read_manifest(out)
        for child in list(out.iterdir()):
            if child.name not in _PAYLOAD:
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        content = content_sha256(out)
        if parent_manifest is not None:
            if parent_manifest.get("content_sha256") != content:
                raise BundleError("원본 번들의 manifest가 자기 내용과 맞지 않습니다")
            parent = {"library_version": parent_manifest["library_version"],
                      "content_sha256": content}
        else:
            parent = {"library_version": parent_version, "content_sha256": content}
    elif source.is_dir():
        for name in _PAYLOAD:
            origin = source / name
            if origin.is_dir():
                shutil.copytree(origin, out / name, ignore=shutil.ignore_patterns(*SKIP_NAMES))
            elif origin.is_file():
                shutil.copy2(origin, out / name)
        content = content_sha256(out)
        parent = _parent_of(source, parent_version)
        if parent["content_sha256"] != content:
            raise BundleError("복사한 내용이 원본과 다릅니다(원본 번들이 바뀌는 중일 수 있습니다)")
    else:
        raise BundleError(f"번들 폴더나 .tar.gz가 아닙니다: {source}")

    curation = {"mode": "baseline",
                "source_sha256": sha256_file(source) if source.is_file() else None}
    return _finish(out, parent=parent, curation=curation,
                   replay_gate={"status": "not_required",
                                "reason": "baseline: content identical to parent"},
                   compat={"status": "passed", "excluded": []}, content=content)


# ── curated ───────────────────────────────────────────────────────────
class _BatchRecords:
    """`<curation>/batches/<batch_id>/manifest.json`의 포즈 레코드를 필요할 때만 읽는다."""

    def __init__(self, curation_dir: Path):
        self.root = curation_dir / "batches"
        self._cache: dict[str, tuple[Path, dict[str, dict]]] = {}

    def get(self, batch_id: str, pose_id: str) -> tuple[Path, dict]:
        if batch_id not in self._cache:
            path = self.root / batch_id / "manifest.json"
            if not path.is_file():
                raise BundleError(f"배치 manifest가 없습니다: {batch_id}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1:
                raise BundleError(f"지원하지 않는 배치 manifest 형식: {batch_id}")
            records = {record["pose_id"]: record for record in payload.get("poses", [])}
            self._cache[batch_id] = (path.parent, records)
        batch_dir, records = self._cache[batch_id]
        if pose_id not in records:
            raise BundleError(f"배치 {batch_id}에 포즈 레코드가 없습니다")
        return batch_dir, records[pose_id]


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise BundleError("배치 manifest의 경로가 배치 폴더 밖을 가리킵니다")
    return path


def _resize_thumbnail(source: Path, target: Path) -> None:
    from PIL import Image

    with Image.open(source) as image:
        if image.width != image.height:
            raise BundleError(f"정사각형이 아닌 미리보기는 줄이지 않습니다: {source.name}")
        image.convert("RGB").resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS).save(
            target, "JPEG", quality=THUMB_QUALITY, optimize=True)


_RESULT_FIELDS = ("filename", "content_type", "width", "height", "size_bytes", "sha256",
                  "artifact_source", "status")


def _existing_result(base_entry: dict | None, path: Path, pose_id: str, view: str) -> dict:
    """기존 썸네일은 바이트 그대로 복사하므로 기준 manifest 항목을 재사용한다.

    항목이 불완전하거나 크기가 다르면 파일로 다시 잰다(출처 표기는 유지).
    """
    if (base_entry and all(field in base_entry for field in _RESULT_FIELDS)
            and base_entry["size_bytes"] == path.stat().st_size):
        return dict(base_entry)
    source = (base_entry or {}).get("artifact_source") or "existing_bundle"
    return _thumb_result(path, pose_id, view, source)


def _thumb_result(path: Path, pose_id: str, view: str, artifact_source: str) -> dict:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return {"pose_id": pose_id, "view": view, "filename": path.name,
            "content_type": "image/jpeg", "width": width, "height": height,
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            "artifact_source": artifact_source, "status": "ok", "error": ""}


def _write_thumbnail_manifest(out: Path, results: list[dict], poses: int,
                              base_manifest: dict | None) -> None:
    total = sum(r["size_bytes"] for r in results)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "asset_role": "service_thumbnail",
        "views": list(VIEW_ORDER),
        "format": "jpeg",
        "extension": ".jpg",
        "content_type": "image/jpeg",
        "width": THUMB_SIZE,
        "height": THUMB_SIZE,
        "quality": THUMB_QUALITY,
        "counts": {"poses": poses, "views_per_pose": len(VIEW_ORDER),
                   "expected": poses * len(VIEW_ORDER), "ok": len(results), "error": 0,
                   "by_artifact_source": dict(Counter(r["artifact_source"] for r in results))},
        "total_bytes": total,
        "average_bytes": round(total / len(results), 2) if results else 0,
        "source_build_manifest_sha256": (base_manifest or {}).get("source_build_manifest_sha256"),
        "results": sorted(results, key=lambda r: (r["pose_id"], VIEW_ORDER.index(r["view"]))),
    }
    (out / "thumbs" / "thumbnail_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_attribution(out: Path, base: Path, sources: Counter) -> None:
    parts = []
    if (base / "ATTRIBUTION.md").is_file():
        parts.append((base / "ATTRIBUTION.md").read_text(encoding="utf-8").rstrip() + "\n")
    if sources:
        lines = ["", "## 정리 라이브러리 추가분", "",
                 "아래 출처의 프레임·리그로 만든 포즈가 들어 있다. 배포 시 이 표기를 유지한다.", "",
                 "| 출처 | 저작자 | 라이선스 | 원본 | 포즈 수 |", "|---|---|---|---|---|"]
        for (source, author, license_name, license_url, source_url), count in sorted(
                sources.items(), key=lambda item: tuple(str(v) for v in item[0])):
            licensed = f"[{license_name}]({license_url})" if license_url else str(license_name)
            lines.append(f"| {source} | {author or '-'} | {licensed} | {source_url or '-'} | {count} |")
        parts.append("\n".join(lines) + "\n")
    if parts:
        (out / "ATTRIBUTION.md").write_text("\n".join(parts), encoding="utf-8")


def build_curated(curated_db: Path, curation_dir: Path, base_source: Path, out: Path, *,
                  parent_version: str = LEGACY_VERSION, exclude_unresolved_rigs: bool = False,
                  exclude_private_identifiers: bool = False, force: bool = False) -> dict:
    """정리 DB → 배포 번들. 기존 포즈는 기준 번들의 BVH·썸네일을, 신규 포즈는 배치 산출물을 쓴다."""
    if not curated_db.is_file():
        raise BundleError(f"정리 DB가 없습니다: {curated_db}")
    batches = _BatchRecords(curation_dir)
    with tempfile.TemporaryDirectory() as tmp:
        base = _materialize(base_source, Path(tmp))
        _prepare_out(out, force)
        (out / "bvh").mkdir()
        (out / "thumbs").mkdir()
        try:
            source = sqlite3.connect(curated_db.resolve().as_uri() + "?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise BundleError(f"정리 DB를 열 수 없습니다(검수 서버가 잡고 있는지 확인): {exc}") from exc
        target = sqlite3.connect(out / "poses.db")
        try:
            source.backup(target)
        finally:
            source.close()
        target.row_factory = sqlite3.Row
        # 지운 meta 값이 빈 페이지에 남지 않게 한다(아래 VACUUM과 함께).
        target.execute("PRAGMA secure_delete = ON")

        base_thumbs = None
        if (base / "thumbs" / "thumbnail_manifest.json").is_file():
            base_thumbs = json.loads(
                (base / "thumbs" / "thumbnail_manifest.json").read_text(encoding="utf-8"))
        base_results = {(r["pose_id"], r["view"]): r
                        for r in (base_thumbs or {}).get("results", [])}

        results: list[dict] = []
        excluded: list[dict] = []
        sources: Counter = Counter()
        names: dict[str, str] = {}
        groups: Counter = Counter()
        rows = list(target.execute("SELECT rowid, pose_id, bvh_path, meta_json FROM poses"))
        for row in rows:
            pose_id = row["pose_id"]
            meta = json.loads(row["meta_json"] or "{}")
            if privacy_findings(pose_id):
                if not exclude_private_identifiers:
                    raise BundleError(f"pose_id에 사용자 식별자로 보이는 값이 있습니다(rowid {row['rowid']}). "
                                      "정리 단계에서 포즈를 제외하거나 ID를 바꾸세요")
                # Never copy the private pose ID, BVH, thumbnail or metadata
                # into the release, including its exclusion report.
                excluded.append({"source_rowid": row["rowid"], "reason": "private_identifier",
                                 "group": meta.get("curation_group"), "batch_id": meta.get("batch_id")})
                target.execute("DELETE FROM pose_projections WHERE pose_id=?", (pose_id,))
                target.execute("DELETE FROM poses WHERE pose_id=?", (pose_id,))
                continue
            group = meta.get("curation_group")
            if group == "existing":
                name = Path(str(row["bvh_path"]).replace("\\", "/")).name
                bvh_from = base / "bvh" / name
            elif group == "new":
                name = f"{pose_id}.bvh"
                bvh_from = Path(row["bvh_path"])
            else:
                raise BundleError(f"curation_group이 없거나 알 수 없는 행입니다(rowid {row['rowid']})")
            if not SAFE_STEM.fullmatch(name[:-4]) or not name.endswith(".bvh"):
                raise BundleError(f"번들 파일 이름으로 쓸 수 없는 BVH 이름입니다(rowid {row['rowid']})")
            if name in names:
                raise BundleError(f"BVH 파일 이름이 겹칩니다: {names[name]} / {pose_id}")
            if not bvh_from.is_file():
                raise BundleError(f"BVH가 없습니다: {pose_id}")
            expected = meta.get("bvh_sha256")
            if expected and sha256_file(bvh_from) != expected:
                raise BundleError(f"검수한 뒤 BVH가 바뀌었습니다: {pose_id}")
            try:
                profile = converter_profile(bvh_from)
            except (OSError, ValueError):
                if not exclude_unresolved_rigs:
                    raise BundleError(
                        f"운영 FBX 변환기가 모르는 리그입니다: {pose_id} "
                        "(빼고 만들려면 --exclude-unresolved-rigs)") from None
                excluded.append({"pose_id": pose_id, "reason": "converter_profile_unresolved",
                                 "group": group, "batch_id": meta.get("batch_id")})
                target.execute("DELETE FROM pose_projections WHERE pose_id=?", (pose_id,))
                target.execute("DELETE FROM poses WHERE pose_id=?", (pose_id,))
                continue
            names[name] = pose_id
            groups[(group, profile)] += 1
            shutil.copy2(bvh_from, out / "bvh" / name)

            for view in VIEW_ORDER:
                destination = out / "thumbs" / thumbnail_filename(pose_id, view)
                if group == "existing":
                    origin = base / "thumbs" / thumbnail_filename(pose_id, view)
                    if not origin.is_file():
                        raise BundleError(f"기준 번들에 썸네일이 없습니다: {pose_id}/{view}")
                    shutil.copy2(origin, destination)
                    results.append(_existing_result(base_results.get((pose_id, view)),
                                                    destination, pose_id, view))
                else:
                    batch_dir, record = batches.get(str(meta.get("batch_id")), pose_id)
                    item = (record.get("thumbnails") or {}).get(view)
                    if not item:
                        raise BundleError(f"신규 포즈의 {view} 미리보기가 없습니다: {pose_id}")
                    origin = _contained(batch_dir, item["path"])
                    if sha256_file(origin) != item["sha256"]:
                        raise BundleError(f"렌더 뒤 미리보기가 바뀌었습니다: {pose_id}/{view}")
                    _resize_thumbnail(origin, destination)
                    results.append(_thumb_result(destination, pose_id, view, NEW_THUMB_SOURCE))

            if group == "new":
                sources[(meta.get("source"), meta.get("author"), meta.get("license"),
                         meta.get("license_url"), meta.get("source_url"))] += 1
            target.execute(
                "UPDATE poses SET bvh_path=?, meta_json=? WHERE pose_id=?",
                (f"data/bvh/{name}",
                 json.dumps(sanitize_meta(meta), ensure_ascii=False, sort_keys=True), pose_id))
        target.commit()
        target.execute("VACUUM")
        poses = target.execute("SELECT COUNT(*) FROM poses").fetchone()[0]
        target.close()

        if (base / "thumbs" / "thumbnail_manifest.json").is_file() or results:
            _write_thumbnail_manifest(out, results, poses, base_thumbs)
        _write_attribution(out, base, sources)
        parent = _parent_of(base, parent_version)

    curated_manifest = curated_db.parent / "manifest.json"
    curation = {
        "mode": "curated",
        "curated_db_sha256": sha256_file(curated_db),
        "curated_manifest_sha256": (sha256_file(curated_manifest)
                                    if curated_manifest.is_file() else None),
        "by_group_and_rig": {f"{g}:{p}": n for (g, p), n in sorted(groups.items())},
    }
    compat = {"status": "passed", "excluded": excluded}
    return _finish(out, parent=parent, curation=curation,
                   replay_gate={"status": "not_run"}, compat=compat)


# ── record-gate ───────────────────────────────────────────────────────
def record_gate(bundle: Path, report_path: Path) -> dict:
    """재생 게이트 보고서(JSON)의 결과를 manifest에 남긴다. 번들 내용·버전은 바뀌지 않는다."""
    manifest = verify_manifest(bundle)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = report.get("status")
    if status not in ("passed", "failed"):
        raise BundleError(f"게이트 보고서의 status는 passed/failed여야 합니다: {status!r}")
    manifest["replay_gate"] = {
        "status": status,
        "report_sha256": sha256_file(report_path),
        "thresholds_version": report.get("thresholds_version"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "summary": report.get("summary", {}),
    }
    write_manifest(bundle, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="포즈 라이브러리 배포 번들과 library_manifest.json을 만든다.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    baseline = sub.add_parser("baseline", help="기존 번들 + manifest(내용 불변)")
    baseline.add_argument("--source", required=True, help="번들 폴더 또는 .tar.gz")
    baseline.add_argument("--out", required=True)
    baseline.add_argument("--parent-version",
                          default=os.getenv("POSE_LIBRARY_VERSION", LEGACY_VERSION))
    baseline.add_argument("--force", action="store_true", help="출력 폴더를 비우고 다시 만든다")

    curated = sub.add_parser("curated", help="정리 DB → 배포 번들")
    curated.add_argument("--curated-db", required=True)
    curated.add_argument("--curation-dir", required=True,
                         help="batches/<id>/manifest.json이 있는 정리 폴더")
    curated.add_argument("--base", required=True, help="기존 포즈의 BVH·썸네일을 가져올 기준 번들")
    curated.add_argument("--out", required=True)
    curated.add_argument("--parent-version",
                         default=os.getenv("POSE_LIBRARY_VERSION", LEGACY_VERSION))
    curated.add_argument("--exclude-unresolved-rigs", action="store_true",
                         help="운영 변환기가 모르는 리그의 포즈를 빼고 만든다(manifest에 기록)")
    curated.add_argument("--exclude-private-identifiers", action="store_true",
                         help="사용자 식별자를 포함한 포즈를 로컬 검수에만 남기고 번들에서는 제외")
    curated.add_argument("--force", action="store_true")

    gate = sub.add_parser("record-gate", help="재생 게이트 결과를 manifest에 기록")
    gate.add_argument("--bundle", required=True)
    gate.add_argument("--report", required=True)

    args = parser.parse_args()
    try:
        if args.command == "baseline":
            out = Path(args.out)
            _print_done(out, build_baseline(Path(args.source), out,
                                            parent_version=args.parent_version, force=args.force))
        elif args.command == "curated":
            out = Path(args.out)
            manifest = build_curated(
                Path(args.curated_db), Path(args.curation_dir), Path(args.base), out,
                parent_version=args.parent_version,
                exclude_unresolved_rigs=args.exclude_unresolved_rigs,
                exclude_private_identifiers=args.exclude_private_identifiers,
                force=args.force)
            _print_done(out, manifest)
            if manifest["compat"]["excluded"]:
                print(f"  변환기가 모르는 리그로 뺀 포즈 {len(manifest['compat']['excluded'])}개"
                      " (manifest.compat.excluded)")
        else:
            manifest = record_gate(Path(args.bundle), Path(args.report))
            print(f"replay_gate = {manifest['replay_gate']['status']} "
                  f"({manifest['library_version']})")
    except (BundleError, LibraryManifestError) as exc:
        sys.stdout.flush()
        print(f"\n중단합니다. {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
