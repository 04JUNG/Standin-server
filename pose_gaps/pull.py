"""BFF 공백 관측 export를 받아 로컬 스냅샷 하나로 만든다.

    STANDIN_ADMIN_TOKEN=… python -m pose_gaps pull --base-url https://api.standinpose.com --days 90

- 토큰은 환경 변수에서만 읽고 디스크에 쓰지 않는다.
- 모든 항목을 계약(`observations.parse_observation`)으로 검사한 뒤에만 스냅샷을 만든다.
  원본 ID 모양 값이나 금지 키가 하나라도 있으면 아무것도 쓰지 않는다.
- 성공하면 이전 스냅샷을 지우고 최신 하나만 남긴다.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Callable
from urllib.parse import urlencode
import urllib.request

from .observations import ObservationError, check_page, dump_jsonl, parse_observation
from .ttl import EXPORT_META, OBSERVATIONS, SNAPSHOTS, list_snapshots, utc_now

TOKEN_ENV = "STANDIN_ADMIN_TOKEN"
ENDPOINT = "/v1/admin/gaps/observations"
MAX_PAGES = 1000


class PullError(RuntimeError):
    """export를 받지 못했거나 계약과 달랐다. 스냅샷은 만들어지지 않는다."""


def fetch_json(url: str, token: str, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(url, headers={"X-Beta-Admin-Token": token,
                                                   "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https 운영 BFF
        return json.loads(response.read().decode("utf-8"))


def pull(base_url: str, *, days: int, root: Path, token: str | None = None,
         now: datetime | None = None,
         fetch: Callable[[str, str], dict] = fetch_json) -> Path:
    token = token or os.getenv(TOKEN_ENV)
    if not token:
        raise PullError(f"{TOKEN_ENV} 환경 변수가 필요합니다")
    if not 1 <= days <= 365:
        raise PullError("days는 1~365여야 합니다(연결 가능한 데이터의 보관 기한)")
    now = now or utc_now()

    items, export_id, header, cursor = [], None, None, None
    for _ in range(MAX_PAGES):
        query = {"days": days, **({"cursor": cursor} if cursor else {})}
        page = fetch(f"{base_url.rstrip('/')}{ENDPOINT}?{urlencode(query)}", token)
        try:
            raw_items = check_page(page)
            for raw in raw_items:
                parse_observation(raw)
        except ObservationError as exc:
            raise PullError(f"export가 계약과 다릅니다: {exc}") from exc
        if export_id is None:
            export_id, header = page.get("exportId"), page
        elif page.get("exportId") != export_id:
            raise PullError("페이지 사이에 exportId가 바뀌었습니다. 처음부터 다시 받으세요")
        items.extend(raw_items)
        cursor = page.get("nextCursor")
        if not cursor:
            break
    else:
        raise PullError(f"페이지가 {MAX_PAGES}개를 넘었습니다")

    base = Path(root) / SNAPSHOTS
    base.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    temporary = base / f".tmp-{stamp}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    count = dump_jsonl(temporary / OBSERVATIONS, items)
    digest = hashlib.sha256((temporary / OBSERVATIONS).read_bytes()).hexdigest()
    meta = {
        "schema_version": 1,
        "export_id": export_id,
        "window": header.get("window"),
        "generated_at": header.get("generatedAt"),
        "retention": header.get("retention"),
        "pulled_at": now.isoformat(),
        "count": count,
        "observations_sha256": digest,
        "source": f"{base_url.rstrip('/')}{ENDPOINT}",
    }
    (temporary / EXPORT_META).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    final = base / stamp
    if final.exists():
        shutil.rmtree(final)
    temporary.rename(final)
    for _, older in list_snapshots(root):
        if older != final:
            shutil.rmtree(older)
    return final
