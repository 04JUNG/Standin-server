"""라이브러리 산출물이 있어야만 도는 테스트를 표시한다.

`data/`는 Mixamo·CMU 원본 재배포 금지 정책으로 레포에 커밋하지 않는다(.gitignore).
따라서 CI 체크아웃에는 존재하지 않는다. 이 가드를 쓴 테스트는 라이브러리·semantic
빌드를 내려받은 로컬·평가 환경에서만 실행되고, CI에서는 사유와 함께 skip된다.

가드를 쓰지 않고 `data/`를 직접 읽으면 CI에서 실패하거나(수집 단계면) 전체 실행이
중단된다. 새 테스트가 산출물을 읽는다면 이 함수를 먼저 호출한다.
"""
from __future__ import annotations

from pathlib import Path
from unittest import SkipTest

REPO_ROOT = Path(__file__).resolve().parents[1]


def require_library_data(*relative_paths: str) -> None:
    """지정한 산출물이 없으면 skip한다. 인자가 없으면 `data/semantic`을 본다."""
    targets = relative_paths or ("data/semantic",)
    missing = [p for p in targets if not (REPO_ROOT / p).exists()]
    if missing:
        raise SkipTest("library data unavailable: " + ", ".join(missing))
