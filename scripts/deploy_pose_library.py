"""포즈 라이브러리 배포 — 검증 → 압축 → 업로드 → 추론 서비스 재기동 → 확인.

추론 서버 담당자가 명령 하나로 끝내기 위한 스크립트.

    python scripts/deploy_pose_library.py data/          # 배포
    python scripts/deploy_pose_library.py data/ -n       # 검증만(업로드하지 않음)
    python scripts/deploy_pose_library.py --rollback     # 직전 번들로 되돌리기

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

from src.repo import FEATURE_VERSION
from src.schema import View
from src.thumbnails import THUMBNAIL_VIEWS

# ── 배포 대상. 관리자가 알려준 값이 다르면 env로 덮어쓴다 ──────────────
BUCKET = os.getenv("POSE_LIBRARY_BUCKET", "standinapp-assetsbucket5cb76180-rhs7xpvmvhbo")
KEY = os.getenv("POSE_LIBRARY_KEY", "pose-library/v1.tar.gz")  # 서버가 읽는 고정 경로
CLUSTER = os.getenv("ECS_CLUSTER", "StandinApp-ClusterEB0386A7-YtBcZrnPfn06")
SERVICE = os.getenv("ECS_SERVICE", "StandinApp-InferenceService1C7A7625-KPZkcW87EjUE")
REGION = os.getenv("AWS_REGION", "ap-northeast-2")
PROFILE = os.getenv("AWS_PROFILE", "standin-inference")

# 번들 루트에 담는 것. data/ 안의 다른 파일이 딸려 올라가지 않도록 명시적으로 고정한다.
BUNDLE_DB = "poses.db"
BUNDLE_OPTIONAL = ["index.pkl"]          # 없으면 첫 기동에 재계산된다
BUNDLE_DIRS = ["bvh", "thumbs"]
_SKIP_NAMES = {"__pycache__", ".DS_Store", "Thumbs.db"}


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
def validate(data_dir: Path, allow_missing_thumbs: bool = False) -> dict:
    """서버가 기동 시 실제로 요구하는 조건만 검사한다.

    각 항목은 서버 코드의 어느 지점이 그것을 강제하는지 대응된다. 여기서 막지 못하면
    번들이 S3에 올라간 뒤 태스크 기동 시점에야 실패하는데, 번들은 태스크 정의 밖에 있어서
    ECS circuit breaker 롤백으로도 되돌릴 수 없다.
    """
    db_path = data_dir / BUNDLE_DB
    if not db_path.is_file():
        raise DeployError(
            f"{db_path} 가 없습니다. 번들 루트에 poses.db가 있어야 합니다.\n"
            f"  → 만들기: BVH_DIR={data_dir}/bvh python scripts/build_db.py"
        )

    # 읽기 전용으로 연다. repo.connect()는 없는 테이블을 만들어 버려서 검증을 무력화한다.
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return _validate_db(con, data_dir, allow_missing_thumbs)
    finally:
        con.close()


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

    if problems:
        raise DeployError("번들 검증 실패:\n  - " + "\n  - ".join(problems))
    return {"poses": len(poses), "projections": n_proj}


def _check_bvh(poses: list, data_dir: Path) -> list[str]:
    """bvh_path는 DB에 'data/bvh/<name>.bvh'로 들어 있고, 컨테이너 WORKDIR(/app)
    기준 상대경로로 해석된다(DATA_DIR=/app/data). 파일이 없으면 /pose/{id}/bvh가 404."""
    missing = []
    for p in poses:
        raw = (p["bvh_path"] or "").replace("\\", "/")
        if not raw:
            missing.append(p["pose_id"])
            continue
        rel = raw.split("data/", 1)[-1] if "data/" in raw else raw
        if not (data_dir / rel).is_file():
            missing.append(p["pose_id"])
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
    """thumbs/<pose_id>__<view>.png — 없으면 thumbnails.thumbnail_url이 None을 돌려주고
    썸네일만 조용히 사라진다(에러가 안 난다). 그래서 기본값을 '실패'로 둔다."""
    thumbs_dir = data_dir / "thumbs"
    wanted = [(r[0], r[1]) for r in con.execute(
        "SELECT pose_id, view FROM pose_projections") if r[1] in THUMBNAIL_VIEWS]
    missing = [f"{pid}__{view}" for pid, view in wanted
               if not (thumbs_dir / f"{pid}__{view}.png").is_file()]
    if not missing:
        _ok("thumbs", f"{len(wanted)}개 · 누락 0")
        return []
    if allow_missing:
        print(f"      {_pad('thumbs', 16)} {_pad(f'누락 {len(missing)}개', 38)} SKIP")
        return []
    _fail("thumbs", f"누락 {len(missing)}개 / {len(wanted)}")
    return [f"썸네일 {len(missing)}개 누락: {', '.join(missing[:3])} … "
            f"(에러 없이 썸네일만 사라집니다. 의도한 것이면 --allow-missing-thumbs)"]


# ── 2. 압축 ────────────────────────────────────────────────────────────
def make_archive(data_dir: Path, out_path: Path) -> int:
    """번들 루트에 poses.db · index.pkl · bvh/ · thumbs/ 를 담는다(`tar -C data`와 동일).

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


# ── 명령 ───────────────────────────────────────────────────────────────
def cmd_deploy(args) -> None:
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise DeployError(f"폴더가 아닙니다: {data_dir}")

    print(f"[1/5] 번들 검증              ({data_dir})")
    summary = validate(data_dir, args.allow_missing_thumbs)

    if args.dry_run:
        print(f"\n검증 통과 — 포즈 {summary['poses']}개 / 투영 {summary['projections']}개.")
        print("  --dry-run이라 업로드하지 않았습니다. 실행 중인 서비스는 그대로입니다.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "v1.tar.gz"
        size = make_archive(data_dir, archive)
        print(f"[2/5] 압축                   {size / 1024 / 1024:.1f} MiB")

        s3, ecs = _clients()
        prev = _current_version(s3)
        s3.upload_file(str(archive), BUCKET, KEY)
        print(f"[3/5] 업로드                 s3://{BUCKET}/{KEY}")

    restart_and_wait(ecs, args.timeout_min, "4/5", "5/5")
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
    restart_and_wait(ecs, args.timeout_min, "3/4", "4/4")
    print("\n롤백 완료 — 새 태스크가 헬스체크를 통과했습니다.")


def main() -> int:
    p = argparse.ArgumentParser(
        description="포즈 라이브러리를 검증하고 추론 서버에 배포한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예시:\n"
               "  python scripts/deploy_pose_library.py data/\n"
               "  python scripts/deploy_pose_library.py data/ --dry-run\n"
               "  python scripts/deploy_pose_library.py --rollback\n",
    )
    p.add_argument("data_dir", nargs="?", default="data",
                   help="poses.db · bvh/ · thumbs/ 가 있는 폴더 (기본: data)")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="검증만 하고 업로드하지 않는다")
    p.add_argument("--rollback", action="store_true",
                   help="S3의 직전 번들로 되돌리고 재기동한다")
    p.add_argument("--allow-missing-thumbs", action="store_true",
                   help="썸네일 누락을 실패로 보지 않는다(썸네일이 조용히 사라집니다)")
    p.add_argument("--timeout-min", type=int, default=10,
                   help="안정화 대기 시간(분, 기본 10)")
    p.add_argument("-y", "--yes", action="store_true", help="롤백 확인 프롬프트 생략")
    args = p.parse_args()

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
