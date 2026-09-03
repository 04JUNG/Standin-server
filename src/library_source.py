"""포즈 라이브러리 프로비저닝.

라이브러리(poses.db · index.pkl · bvh/ · thumbs/)는 이미지에 넣을 수 없다:
  · Mixamo/CMU 재배포 금지 조항 → .gitignore로 커밋 차단
  · .dockerignore로 이미지에서도 제외

그래서 배포 환경에서는 기동 시 외부(S3 등)에서 번들을 받아 푼다.
로컬 개발은 기존처럼 data/에 직접 두면 되고, 이 모듈은 아무 것도 하지 않는다.

번들 형식: tar.gz. 루트에 아래를 담는다.
    poses.db
    index.pkl                    (선택 — 없으면 첫 기동에 재계산)
    bvh/*.bvh                    (선택 — /pose/{id}/bvh 응답에 필요)
    thumbs/<id>__<view>.png      (선택 — 빠지면 썸네일이 조용히 사라진다)
    thumbs/thumbnail_manifest.json

만드는 법 — 검증까지 해주는 배포 스크립트를 쓴다:
    python scripts/render_bvh_thumbnails.py data/bvh data/thumbs
    python scripts/deploy_pose_library.py data/

직접 만들 때(thumbs를 빠뜨리지 않는다):
    tar -czf pose-library-v1.tar.gz -C data poses.db index.pkl bvh thumbs
    aws s3 cp pose-library-v1.tar.gz s3://<bucket>/pose-library/v1.tar.gz
"""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


class LibraryFetchError(RuntimeError):
    """번들을 가져오지 못했다. 기동을 중단시키는 용도."""


def _download_s3(uri: str, dest: Path) -> None:
    # boto3는 배포 환경에서만 필요하다(로컬 개발에 강제하지 않음).
    try:
        import boto3  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover - 환경 의존
        raise LibraryFetchError(
            f"{uri} 를 받으려면 boto3가 필요합니다. requirements에 boto3를 추가하세요."
        ) from e

    without_scheme = uri[len("s3://") :]
    if "/" not in without_scheme:
        raise LibraryFetchError(f"S3 URI 형식이 잘못됐습니다: {uri}")
    bucket, key = without_scheme.split("/", 1)

    # 자격증명은 ECS 태스크 역할에서 자동으로 온다(키를 코드/환경에 두지 않는다).
    boto3.client("s3").download_file(bucket, key, str(dest))


def _download_http(uri: str, dest: Path) -> None:
    with urllib.request.urlopen(uri) as res, dest.open("wb") as f:  # noqa: S310
        shutil.copyfileobj(res, f)


def _safe_extract(archive: Path, dest_dir: Path) -> None:
    """경로 탈출(../, 절대경로)을 막고 푼다."""
    dest_dir = dest_dir.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest_dir / member.name).resolve()
            if not str(target).startswith(str(dest_dir)):
                raise LibraryFetchError(f"번들에 비정상 경로가 있습니다: {member.name}")
        tar.extractall(dest_dir)  # noqa: S202 - 위에서 경로 검증함


def ensure_library(data_dir: str, db_path: str, uri: str | None) -> bool:
    """라이브러리를 준비한다.

    Returns:
        True  — 번들을 새로 받아 풀었다.
        False — 이미 있거나 uri가 없어 아무 것도 하지 않았다.
    """
    if os.path.exists(db_path):
        return False  # 이미 있다(로컬 개발 또는 이전 기동에서 받아둠)
    if not uri:
        return False  # 받을 곳이 지정되지 않음 → 호출부가 정책을 결정

    dest = Path(data_dir)
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "pose-library.tar.gz"
        try:
            if uri.startswith("s3://"):
                _download_s3(uri, archive)
            elif uri.startswith(("http://", "https://")):
                _download_http(uri, archive)
            else:
                raise LibraryFetchError(
                    f"지원하지 않는 POSE_LIBRARY_URI 형식입니다: {uri} (s3:// 또는 https://)"
                )
            _safe_extract(archive, dest)
        except LibraryFetchError:
            raise
        except Exception as e:
            raise LibraryFetchError(f"포즈 라이브러리를 가져오지 못했습니다({uri}): {e}") from e

    if not os.path.exists(db_path):
        raise LibraryFetchError(
            f"번들을 풀었지만 {db_path} 가 없습니다. 번들 루트에 poses.db가 있어야 합니다."
        )
    return True
