"""포즈 라이브러리 배포 — 검증 → 압축 → 업로드 → 추론 서비스 재기동 → 확인.

추론 서버 담당자가 명령 하나로 끝내기 위한 스크립트.

    python scripts/build_pose_bundle.py baseline \
        --source data/_s3/pose-library-v1.tar.gz --out data/bundles/next
    python scripts/deploy_pose_library.py data/bundles/next      # 배포
    python scripts/deploy_pose_library.py data/bundles/next -n   # 검증만(업로드하지 않음)
    python scripts/deploy_pose_library.py --rollback     # 직전 번들로 되돌리기

번들에는 `library_manifest.json`(내용 해시에서 만든 `lib-YYYYMMDD-<hash8>` 버전)이 있어야
한다. S3 key가 고정이라 이것 없이는 어느 번들이 답했는지 알 수 없다. manifest 없는 옛
번들을 꼭 올려야 하면 `--allow-no-manifest`를 쓴다. 흐름은 `docs/POSE_LIBRARY_BUNDLE.md`.

왜 이 저장소에 있나:
    검증이 서버 상수(`repo.FEATURE_VERSION` · `schema.View` · `thumbnails.THUMBNAIL_VIEWS`)를
    그대로 import 한다. 규격이 바뀌면 검증도 같이 따라가므로 "검증은 통과했는데 서버는
    거부"가 구조적으로 생기지 않는다. 상수를 복사해 두면 조용히 어긋난다.

권한:
    `standin-inference-operator` 정책만으로 동작한다.
      s3:PutObject · s3:GetObject(Version) · s3:ListBucketVersions(pose-library/*)
      ecs:UpdateService · ecs:DescribeServices
    DescribeTasks·CloudWatch 로그 읽기 권한은 필요하지 않다(아래 '성공 판정').

성공 판정:
    추론 서버 `/healthz`는 `pose_count == 0`이면 503을 반환하고(api/app.py), ECS 컨테이너
    헬스체크가 그 응답으로 태스크를 판정한다. 따라서 **서비스 안정화 성공 = 새 번들이
    실제로 파싱되어 비어 있지 않게 로드됐다는 증거**다. 로그를 따로 뒤질 필요가 없다.

Fargate 컨테이너의 로컬 파일은 태스크 교체 시 사라진다. 서버에 파일을 직접 복사하지 않고,
항상 S3 고정 경로에 올린 뒤 서비스를 재기동한다.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sqlite3
import sys
import tarfile
import tempfile
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows의 파이썬은 콘솔 코드페이지(cp949)로 인코딩하는데 Git Bash는 UTF-8로 읽는다
# → 한글이 깨진다. 이 스크립트는 Git Bash에서 쓰는 것을 전제하므로 UTF-8로 고정한다.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except AttributeError:  # 파이썬 3.6 이하 — 그대로 둔다
            pass

from scripts.pose_bundle_policy import (
    bvh_path_ok,
    converter_profile,
    disallowed_meta_keys,
    privacy_findings,
)
from src.library_manifest import (
    MANIFEST_NAME,
    LibraryManifestError,
    gate_allows_deploy,
    read_manifest,
    verify_manifest,
)
from src.repo import FEATURE_VERSION
from src.schema import View
from src.thumbnails import THUMBNAIL_VIEWS, thumbnail_filename

# ── 배포 대상. 관리자가 알려준 값이 다르면 env로 덮어쓴다 ──────────────
BUCKET = os.getenv("POSE_LIBRARY_BUCKET", "standinapp-assetsbucket5cb76180-rhs7xpvmvhbo")
KEY = os.getenv("POSE_LIBRARY_KEY", "pose-library/v1.tar.gz")  # 서버가 읽는 고정 경로
CLUSTER = os.getenv("ECS_CLUSTER", "StandinApp-ClusterEB0386A7-YtBcZrnPfn06")
SERVICE = os.getenv("ECS_SERVICE", "StandinApp-InferenceService1C7A7625-KPZkcW87EjUE")
REGION = os.getenv("AWS_REGION", "ap-northeast-2")
PROFILE = os.getenv("AWS_PROFILE", "standin-inference")

# 번들 루트에 담는 것. data/ 안의 다른 파일이 딸려 올라가지 않도록 명시적으로 고정한다.
# index.pkl은 넣지 않는다 — 서버는 읽지 않고(scripts/run_demo.py만 쓴다), 남겨 두면
# poses.db와 어긋난 오래된 pickle이 번들에 섞인다.
BUNDLE_DB = "poses.db"
BUNDLE_OPTIONAL = [MANIFEST_NAME, "ATTRIBUTION.md"]
BUNDLE_DIRS = ["bvh", "thumbs"]
_SKIP_NAMES = {"__pycache__", ".DS_Store", "Thumbs.db"}
DEFAULT_LOG = Path("data") / "deploys.jsonl"


class DeployError(RuntimeError):
    """배포를 중단시키는 오류. 스택트레이스 없이 메시지만 보여준다."""


def _pad(text: str, width: int) -> str:
    """한글은 콘솔에서 2칸을 차지한다 → 글자 수가 아니라 표시 폭으로 채운다."""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - shown)


def _ok(label: str, detail: str) -> None:
    print(f"      {_pad(label, 16)} {_pad(detail, 38)} OK")


def _fail(label: str, detail: str) -> None:
    print(f"      {_pad(label, 16)} {_pad(detail, 38)} FAIL")


# ── 1. 검증 ────────────────────────────────────────────────────────────
def validate(data_dir: Path, allow_missing_thumbs: bool = False, *,
             allow_no_manifest: bool = False,
             skip_gate_reason: str | None = None) -> dict:
    """서버가 기동 시 실제로 요구하는 조건과, 번들에 들어가면 안 되는 것을 검사한다.

    각 항목은 서버 코드의 어느 지점이 그것을 강제하는지 대응된다. 여기서 막지 못하면
    번들이 S3에 올라간 뒤 태스크 기동 시점에야 실패하는데, 번들은 태스크 정의 밖에 있어서
    ECS circuit breaker 롤백으로도 되돌릴 수 없다. 개인정보는 더 나쁘다 — 자산 버킷은
    버전 관리라 한 번 올라가면 이전 버전에 남는다.
    """
    db_path = data_dir / BUNDLE_DB
    if not db_path.is_file():
        raise DeployError(
            f"{db_path} 가 없습니다. 번들 루트에 poses.db가 있어야 합니다.\n"
            f"  → 만들기: python scripts/build_pose_bundle.py --help"
        )

    # 읽기 전용으로 연다. repo.connect()는 없는 테이블을 만들어 버려서 검증을 무력화한다.
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        summary = _validate_db(con, data_dir, allow_missing_thumbs)
    finally:
        con.close()
    summary["manifest"] = _check_manifest(data_dir, allow_no_manifest, skip_gate_reason)
    return summary


def _validate_db(con: sqlite3.Connection, data_dir: Path, allow_missing_thumbs: bool) -> dict:
    problems: list[str] = []

    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing_tables = {"poses", "pose_projections"} - tables
    if missing_tables:
        raise DeployError(f"poses.db에 테이블이 없습니다: {', '.join(sorted(missing_tables))}")

    # feature_version — repo.load_entries가 불일치 시 RuntimeError로 기동을 막는다.
    versions = sorted(r[0] for r in con.execute(
        "SELECT DISTINCT feature_version FROM pose_projections"))
    if versions == [FEATURE_VERSION]:
        _ok("feature_version", f"{FEATURE_VERSION} == src/repo.py 규격")
    else:
        _fail("feature_version", f"{versions} != [{FEATURE_VERSION}]")
        problems.append(
            f"feature_version 불일치({versions} != [{FEATURE_VERSION}]). "
            "DB를 재빌드하세요: python scripts/build_db.py")

    # view 값 — repo.load_entries가 View(...)로 변환하므로 모르는 값이면 ValueError.
    known = {v.value for v in View}
    views = sorted(r[0] for r in con.execute("SELECT DISTINCT view FROM pose_projections"))
    unknown = [v for v in views if v not in known]
    if unknown:
        _fail("view", f"알 수 없는 값 {unknown}")
        problems.append(f"schema.View에 없는 view 값: {unknown}")
    else:
        _ok("view", ", ".join(views))

    poses = list(con.execute("SELECT pose_id, bvh_path FROM poses"))
    proj_count = dict(con.execute(
        "SELECT pose_id, COUNT(*) FROM pose_projections GROUP BY pose_id"))
    n_proj = con.execute("SELECT COUNT(*) FROM pose_projections").fetchone()[0]
    if not poses:
        raise DeployError("poses 테이블이 비어 있습니다. 배포하면 서버가 503으로 뜹니다.")

    # 투영이 없는 포즈는 검색 후보에 영원히 안 나온다(load_entries가 JOIN으로 버린다).
    orphan_pose = [p["pose_id"] for p in poses if p["pose_id"] not in proj_count]
    if orphan_pose:
        _fail("포즈", f"투영 없는 포즈 {len(orphan_pose)}개")
        problems.append(f"투영이 없어 검색에 안 잡히는 포즈 {len(orphan_pose)}개: "
                        f"{', '.join(orphan_pose[:3])} …")
    else:
        _ok("포즈", f"{len(poses)}개 · 투영 {n_proj}개")

    # poses에 없는 pose_id의 투영도 JOIN에서 조용히 사라진다.
    known_ids = {p["pose_id"] for p in poses}
    dangling = [pid for pid in proj_count if pid not in known_ids]
    if dangling:
        problems.append(f"poses에 없는 pose_id의 투영 {len(dangling)}건: "
                        f"{', '.join(dangling[:3])} …")

    problems += _check_bvh(poses, data_dir)
    problems += _check_blobs(con)
    problems += _check_thumbs(con, data_dir, allow_missing_thumbs)
    problems += _check_privacy(con)
    problems += _check_compat(poses, data_dir)

    if problems:
        raise DeployError("번들 검증 실패:\n  - " + "\n  - ".join(problems))
    return {"poses": len(poses), "projections": n_proj}


def _bvh_file(data_dir: Path, bvh_path: str | None) -> Path | None:
    raw = (bvh_path or "").replace("\\", "/")
    if not raw:
        return None
    rel = raw.split("data/", 1)[-1] if "data/" in raw else raw
    path = data_dir / rel
    return path if path.is_file() else None


def _check_bvh(poses: list, data_dir: Path) -> list[str]:
    """bvh_path는 DB에 'data/bvh/<name>.bvh'로 들어 있고, 컨테이너 WORKDIR(/app)
    기준 상대경로로 해석된다(DATA_DIR=/app/data). 파일이 없으면 /pose/{id}/bvh가 404."""
    missing = [p["pose_id"] for p in poses if _bvh_file(data_dir, p["bvh_path"]) is None]
    if missing:
        _fail("bvh", f"파일 없음 {len(missing)}개")
        return [f"DB에는 있으나 번들에 없는 bvh {len(missing)}개: "
                f"{', '.join(missing[:3])} … (동원 핸드오프가 404가 됩니다)"]
    _ok("bvh", f"{len(poses)}개 · DB↔파일 누락 0")
    return []


def _check_blobs(con: sqlite3.Connection) -> list[str]:
    """feature_blob은 np.frombuffer(float32)로 복원된다. 길이가 섞이면 kNN이 깨진다."""
    empty = con.execute("SELECT COUNT(*) FROM pose_projections "
                        "WHERE feature_blob IS NULL OR LENGTH(feature_blob)=0").fetchone()[0]
    lengths = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT LENGTH(feature_blob) FROM pose_projections")})
    if empty:
        _fail("feature_blob", f"빈 값 {empty}건")
        return [f"비어 있는 feature_blob {empty}건"]
    if len(lengths) != 1:
        _fail("feature_blob", f"길이 불균일 {lengths}")
        return [f"feature_blob 길이가 섞여 있습니다: {lengths}"]
    if lengths[0] % 4:
        _fail("feature_blob", f"{lengths[0]}B — float32 배수 아님")
        return [f"feature_blob 길이 {lengths[0]}B가 float32(4B) 배수가 아닙니다"]
    _ok("feature_blob", f"빈 값 0 · 길이 균일({lengths[0]}B)")
    return []


def _check_thumbs(con: sqlite3.Connection, data_dir: Path, allow_missing: bool) -> list[str]:
    """thumbs/<pose_id>__<view>.jpg — 없으면 thumbnails.thumbnail_url이 None을 돌려주고
    썸네일만 조용히 사라진다(에러가 안 난다). 그래서 기본값을 '실패'로 둔다.

    파일 이름은 서버와 같은 `thumbnails.thumbnail_filename`으로 만든다. 예전에는 여기만
    `.png`를 찾아서, 2026-09-03 JPEG 번들 이후 정상 번들도 검증에서 막혔다."""
    thumbs_dir = data_dir / "thumbs"
    wanted = [(r[0], r[1]) for r in con.execute(
        "SELECT pose_id, view FROM pose_projections") if r[1] in THUMBNAIL_VIEWS]
    missing = [f"{pid}__{view}" for pid, view in wanted
               if not (thumbs_dir / thumbnail_filename(pid, view)).is_file()]
    if not missing:
        _ok("thumbs", f"{len(wanted)}개 · 누락 0")
        return []
    if allow_missing:
        print(f"      {_pad('thumbs', 16)} {_pad(f'누락 {len(missing)}개', 38)} SKIP")
        return []
    _fail("thumbs", f"누락 {len(missing)}개 / {len(wanted)}")
    return [f"썸네일 {len(missing)}개 누락: {', '.join(missing[:3])} … "
            f"(에러 없이 썸네일만 사라집니다. 의도한 것이면 --allow-missing-thumbs)"]


def _check_privacy(con: sqlite3.Connection) -> list[str]:
    """사용자 식별자·로컬 경로·허용 목록 밖 meta를 막는다(scripts/pose_bundle_policy.py).

    어긋난 값 자체는 출력하지 않는다. 위치는 poses의 rowid로 알려 준다.
    """
    leaks: dict[str, int] = {}
    leak_rows: list[int] = []
    bad_keys: dict[str, int] = {}
    bad_paths: list[int] = []
    for row in con.execute("SELECT rowid, pose_id, bvh_path, meta_json FROM poses"):
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except ValueError:
            meta = {"invalid_meta_json": True}
        if not isinstance(meta, dict):
            meta = {"invalid_meta_json": True}
        found = privacy_findings(row["pose_id"], meta, row["bvh_path"])
        for label in found:
            leaks[label] = leaks.get(label, 0) + 1
        if found:
            leak_rows.append(row["rowid"])
        for key in disallowed_meta_keys(meta):
            bad_keys[key] = bad_keys.get(key, 0) + 1
        if not bvh_path_ok(row["bvh_path"]):
            bad_paths.append(row["rowid"])

    problems = []
    if leaks:
        detail = ", ".join(f"{k} {v}" for k, v in sorted(leaks.items()))
        problems.append(f"사용자 식별자·로컬 경로로 보이는 값 {len(leak_rows)}행({detail}). "
                        f"poses rowid {leak_rows[:5]} … — 번들 빌더로 다시 만드세요")
    if bad_keys:
        detail = ", ".join(f"{k} {v}" for k, v in sorted(bad_keys.items()))
        problems.append(f"허용 목록 밖 meta 키: {detail} — build_pose_bundle.py가 걸러 냅니다")
    if bad_paths:
        problems.append(f"bvh_path가 'data/bvh/<파일>.bvh' 형식이 아닌 행 {len(bad_paths)}개 "
                        f"(rowid {bad_paths[:5]} …)")
    if problems:
        _fail("privacy", f"문제 {len(problems)}종")
    else:
        _ok("privacy", "식별자 0 · meta 허용 목록 · 경로 형식")
    return problems


def _check_compat(poses: list, data_dir: Path) -> list[str]:
    """운영 FBX 변환기가 리그를 알아보는지. 모르면 검색엔 나오고 내보내기는 실패한다."""
    profiles: dict[str, int] = {}
    unresolved: list[str] = []
    for p in poses:
        path = _bvh_file(data_dir, p["bvh_path"])
        if path is None:
            continue  # _check_bvh가 이미 보고했다
        try:
            profile = converter_profile(path)
        except (OSError, ValueError):
            unresolved.append(p["pose_id"])
            continue
        profiles[profile] = profiles.get(profile, 0) + 1
    if unresolved:
        _fail("rig", f"변환기가 모르는 리그 {len(unresolved)}개")
        return [f"운영 FBX 변환기(converter.bone_map)가 모르는 리그 {len(unresolved)}개: "
                f"{', '.join(unresolved[:3])} … (내보내기가 실패합니다)"]
    _ok("rig", ", ".join(f"{k} {v}" for k, v in sorted(profiles.items())) or "-")
    return []


def _check_manifest(data_dir: Path, allow_missing: bool,
                    skip_gate_reason: str | None) -> dict | None:
    """library_manifest.json의 형식·내용 해시와 재생 게이트 결과를 확인한다."""
    try:
        manifest = read_manifest(data_dir)
    except LibraryManifestError as exc:
        _fail("manifest", "읽기 실패")
        raise DeployError(str(exc)) from exc
    if manifest is None:
        if allow_missing:
            print(f"      {_pad('manifest', 16)} {_pad('없음(--allow-no-manifest)', 38)} SKIP")
            return None
        _fail("manifest", "없음")
        raise DeployError(
            f"{data_dir / MANIFEST_NAME} 가 없습니다. 어느 번들이 답했는지 알 수 없게 됩니다.\n"
            "  → 만들기: python scripts/build_pose_bundle.py baseline --help\n"
            "  → 옛 번들을 그대로 올려야 하면: --allow-no-manifest")
    try:
        verify_manifest(data_dir, manifest, check_content=True)
    except LibraryManifestError as exc:
        _fail("manifest", "내용과 불일치")
        raise DeployError(str(exc)) from exc
    _ok("manifest", str(manifest["library_version"]))

    gate = (manifest.get("replay_gate") or {}).get("status", "not_run")
    if gate_allows_deploy(manifest):
        _ok("replay_gate", str(gate))
    elif skip_gate_reason:
        print(f"      {_pad('replay_gate', 16)} {_pad(f'{gate} (--skip-gate)', 38)} SKIP")
    else:
        _fail("replay_gate", str(gate))
        raise DeployError(
            f"재생 게이트가 통과하지 않았습니다(status={gate}). 기존 검색 결과가 나빠지지 "
            "않는지 먼저 확인하세요.\n"
            "  → 근거를 남기고 건너뛰려면: --skip-gate --reason \"…\"")
    return manifest


# ── 2. 압축 ────────────────────────────────────────────────────────────
def make_archive(data_dir: Path, out_path: Path) -> int:
    """번들 루트에 poses.db · library_manifest.json · ATTRIBUTION.md · bvh/ · thumbs/ 를
    담는다(`tar -C data`와 동일). index.pkl은 서버가 읽지 않으므로 넣지 않는다.

    tar CLI 대신 tarfile을 쓴다 — Git Bash에서 'C:/...' 경로를 원격 호스트로 오인하는
    문제와 한글 경로 인코딩을 피할 수 있다.
    """
    def _add(tar: tarfile.TarFile, src: Path, arc: str) -> None:
        if src.name in _SKIP_NAMES:
            return
        if src.is_dir():
            for child in sorted(src.iterdir()):
                _add(tar, child, f"{arc}/{child.name}")
        elif src.is_file():
            tar.add(src, arcname=arc)

    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(data_dir / BUNDLE_DB, arcname=BUNDLE_DB)
        for name in BUNDLE_OPTIONAL:
            if (data_dir / name).is_file():
                tar.add(data_dir / name, arcname=name)
        for name in BUNDLE_DIRS:
            if (data_dir / name).is_dir():
                _add(tar, data_dir / name, name)
    return out_path.stat().st_size


# ── 3~5. AWS ───────────────────────────────────────────────────────────
def _clients():
    try:
        import boto3
    except ImportError as e:
        raise DeployError("boto3가 필요합니다: pip install boto3") from e

    from botocore.exceptions import ProfileNotFound
    try:
        session = boto3.Session(profile_name=PROFILE, region_name=REGION)
        session.get_credentials()
    except ProfileNotFound:
        print(f"      (프로필 '{PROFILE}' 없음 → 기본 자격증명 사용)")
        session = boto3.Session(region_name=REGION)
    return session.client("s3"), session.client("ecs")


def _current_version(s3) -> str | None:
    """지금 올라가 있는 번들의 VersionId. 롤백 안내에 쓴다."""
    try:
        return s3.head_object(Bucket=BUCKET, Key=KEY).get("VersionId")
    except Exception:
        return None


def restart_and_wait(ecs, timeout_min: int, step_restart: str, step_wait: str) -> None:
    """새 태스크를 띄운다. 태스크는 기동하면서 S3의 최신 번들을 내려받는다."""
    ecs.update_service(cluster=CLUSTER, service=SERVICE, forceNewDeployment=True)
    print(f"[{step_restart}] 추론 서비스 재기동      force-new-deployment")

    attempts = max(1, int(timeout_min * 60 / 15))
    print(f"[{step_wait}] 안정화 대기            최대 {timeout_min}분", flush=True)
    started = time.time()
    waiter = ecs.get_waiter("services_stable")
    try:
        waiter.wait(cluster=CLUSTER, services=[SERVICE],
                    WaiterConfig={"Delay": 15, "MaxAttempts": attempts})
    except Exception as e:
        raise DeployError(
            f"안정화에 실패했습니다({e}).\n"
            "  새 태스크가 헬스체크를 통과하지 못했습니다. 이전 태스크가 계속 서비스 중이라\n"
            "  장애는 아니지만, 새 번들은 적용되지 않았습니다.\n"
            "  → 되돌리기: python scripts/deploy_pose_library.py --rollback"
        ) from e
    print(f"      완료                   {int(time.time() - started)}초")


def _report_success(prev_version: str | None) -> None:
    print("\n배포 완료 — 새 태스크가 헬스체크를 통과했습니다.")
    print("  /healthz는 포즈가 0개면 503을 주므로, 통과 = 새 번들이 로드됐다는 뜻입니다.")
    if prev_version:
        print(f"  직전 번들 VersionId: {prev_version}")
    print("  되돌리려면: python scripts/deploy_pose_library.py --rollback")


def append_deploy_log(path: Path, entry: dict) -> None:
    """배포·롤백 한 건을 JSONL 한 줄로 남긴다. 어느 S3 버전이 어느 library_version인지의 기록이다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": datetime.now(timezone.utc).isoformat(), "bucket": BUCKET, "key": KEY, **entry}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── 명령 ───────────────────────────────────────────────────────────────
def cmd_deploy(args) -> None:
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise DeployError(f"폴더가 아닙니다: {data_dir}")

    print(f"[1/5] 번들 검증              ({data_dir})")
    summary = validate(data_dir, args.allow_missing_thumbs,
                       allow_no_manifest=args.allow_no_manifest,
                       skip_gate_reason=args.reason if args.skip_gate else None)
    manifest = summary.get("manifest") or {}
    if (data_dir / "index.pkl").is_file():
        print("      index.pkl은 번들에 넣지 않습니다(서버가 읽지 않음).")

    if args.dry_run:
        print(f"\n검증 통과 — 포즈 {summary['poses']}개 / 투영 {summary['projections']}개"
              f" · {manifest.get('library_version', 'manifest 없음')}.")
        print("  --dry-run이라 업로드하지 않았습니다. 실행 중인 서비스는 그대로입니다.")
        return

    log_entry = {
        "action": "deploy",
        "library_version": manifest.get("library_version"),
        "content_sha256": manifest.get("content_sha256"),
        "skip_gate_reason": args.reason if args.skip_gate else None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "v1.tar.gz"
        size = make_archive(data_dir, archive)
        print(f"[2/5] 압축                   {size / 1024 / 1024:.1f} MiB")

        s3, ecs = _clients()
        prev = _current_version(s3)
        s3.upload_file(str(archive), BUCKET, KEY)
        print(f"[3/5] 업로드                 s3://{BUCKET}/{KEY}")
        log_entry.update(previous_version_id=prev, version_id=_current_version(s3))

    try:
        restart_and_wait(ecs, args.timeout_min, "4/5", "5/5")
    except DeployError:
        append_deploy_log(Path(args.log), {**log_entry, "result": "restart_failed"})
        raise
    append_deploy_log(Path(args.log), {**log_entry, "result": "stable"})
    _report_success(prev)


def cmd_rollback(args) -> None:
    s3, ecs = _clients()
    print("[1/4] 직전 번들 찾기")
    versions = s3.list_object_versions(Bucket=BUCKET, Prefix=KEY).get("Versions", [])
    versions = [v for v in versions if v["Key"] == KEY]
    versions.sort(key=lambda v: v["LastModified"], reverse=True)
    if len(versions) < 2:
        raise DeployError("되돌릴 이전 버전이 없습니다(S3에 버전이 1개뿐입니다).")

    target = versions[1]
    print(f"      → {target['LastModified']:%Y-%m-%d %H:%M} · "
          f"{target['Size'] / 1024 / 1024:.1f} MiB · {target['VersionId']}")
    if not args.yes:
        if input("      이 번들로 되돌립니다. 계속할까요? [y/N] ").strip().lower() != "y":
            print("취소했습니다. 아무 것도 바꾸지 않았습니다.")
            return

    s3.copy_object(Bucket=BUCKET, Key=KEY,
                   CopySource={"Bucket": BUCKET, "Key": KEY, "VersionId": target["VersionId"]})
    print("[2/4] 복원 완료")
    log_entry = {"action": "rollback", "restored_from_version_id": target["VersionId"],
                 "previous_version_id": versions[0]["VersionId"],
                 "version_id": _current_version(s3)}
    try:
        restart_and_wait(ecs, args.timeout_min, "3/4", "4/4")
    except DeployError:
        append_deploy_log(Path(args.log), {**log_entry, "result": "restart_failed"})
        raise
    append_deploy_log(Path(args.log), {**log_entry, "result": "stable"})
    print("\n롤백 완료 — 새 태스크가 헬스체크를 통과했습니다.")
    print("  manifest 없는 옛 번들로 돌아갔다면 서버는 POSE_LIBRARY_VERSION(env)을 보고합니다.")


def main() -> int:
    p = argparse.ArgumentParser(
        description="포즈 라이브러리를 검증하고 추론 서버에 배포한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예시:\n"
               "  python scripts/build_pose_bundle.py baseline \\\n"
               "      --source data/_s3/pose-library-v1.tar.gz --out data/bundles/next\n"
               "  python scripts/deploy_pose_library.py data/bundles/next --dry-run\n"
               "  python scripts/deploy_pose_library.py data/bundles/next\n"
               "  python scripts/deploy_pose_library.py --rollback\n",
    )
    p.add_argument("data_dir", nargs="?", default="data",
                   help="poses.db · library_manifest.json · bvh/ · thumbs/ 가 있는 번들 폴더 "
                        "(기본: data)")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="검증만 하고 업로드하지 않는다")
    p.add_argument("--rollback", action="store_true",
                   help="S3의 직전 번들로 되돌리고 재기동한다")
    p.add_argument("--allow-missing-thumbs", action="store_true",
                   help="썸네일 누락을 실패로 보지 않는다(썸네일이 조용히 사라집니다)")
    p.add_argument("--timeout-min", type=int, default=10,
                   help="안정화 대기 시간(분, 기본 10)")
    p.add_argument("-y", "--yes", action="store_true", help="롤백 확인 프롬프트 생략")
    p.add_argument("--allow-no-manifest", action="store_true",
                   help="library_manifest.json 없는 옛 번들도 올린다(어느 번들이 답했는지 "
                        "기록이 끊깁니다)")
    p.add_argument("--skip-gate", action="store_true",
                   help="재생 게이트를 통과하지 않은 번들도 올린다. --reason이 필요하다")
    p.add_argument("--reason", default=None, help="--skip-gate의 근거(배포 기록에 남는다)")
    p.add_argument("--log", default=str(DEFAULT_LOG),
                   help=f"배포 기록 JSONL 경로(기본: 실행 위치의 {DEFAULT_LOG.as_posix()})")
    args = p.parse_args()
    if args.skip_gate and not (args.reason or "").strip():
        p.error("--skip-gate에는 --reason이 필요합니다")

    try:
        cmd_rollback(args) if args.rollback else cmd_deploy(args)
    except DeployError as e:
        sys.stdout.flush()   # 로그로 리다이렉트해도 검증 결과 뒤에 오도록
        print(f"\n중단합니다. {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n취소했습니다.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
