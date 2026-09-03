"""구조화 로그 단일 소스. 출력 한 줄 = JSON 객체 하나.

스키마 정본: Standin-master-docs/관측성_로그모니터링_알림_2026-08-18.md §4.
BFF(`Standin-app-server/src/log.ts`)와 **같은 스키마**를 낸다. 두 서비스의 로그를
requestId로 이어 붙이려면 형식이 같아야 한다.

print()를 쓰지 않는 이유는 두 가지다.
  1. 뒤에 붙는 집계·알림이 이 한 줄을 파싱한다. 형식이 제각각이면 파싱이 안 된다.
  2. PII 차단을 여기 한 곳에서 강제할 수 있다.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone

from .config import CFG

SERVICE = "inference"

# 요청 단위 컨텍스트. BFF가 X-Request-Id로 넘겨준 값을 미들웨어가 채운다.
# 요청 밖(기동·백그라운드)에서는 None이고, 그때도 로깅은 그대로 동작해야 한다.
request_id_var: ContextVar[str | None] = ContextVar("standin_request_id", default=None)

# 값이 아니라 **키 이름**으로 막는다. 토큰처럼 생긴 문자열을 정규식으로 잡으려 하면
# 반드시 새는 경로가 생기지만, 이름으로 막으면 새 필드가 자동으로 걸린다.
_REDACT_KEY = re.compile(
    r"token|password|passwd|secret|authorization|cookie|apikey|api_key|credential|email",
    re.IGNORECASE,
)
_REDACTED = "[redacted]"
_MAX_STRING = 512
_MAX_ITEMS = 20
_MAX_DEPTH = 3
_MAX_STACK = 2000

_configured = False


def _sanitize_value(value: object, depth: int) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING else value[:_MAX_STRING] + "…"
    if depth >= _MAX_DEPTH:
        return "[truncated]"
    if isinstance(value, dict):
        return _sanitize(value, depth + 1)
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(item, depth + 1) for item in list(value)[:_MAX_ITEMS]]
    return _sanitize_value(str(value), depth)


def _sanitize(fields: dict, depth: int = 0) -> dict:
    output: dict = {}
    for key, value in fields.items():
        if value is None:
            continue
        if _REDACT_KEY.search(str(key)):
            output[key] = _REDACTED
            continue
        output[key] = _sanitize_value(value, depth)
    return output


class JsonFormatter(logging.Formatter):
    """LogRecord → 공통 스키마 JSON 한 줄."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "service": SERVICE,
            "version": CFG.deployment_version,
            # event_type 없이 들어온 로그(라이브러리·uvicorn)는 `log`로 모은다.
            "type": getattr(record, "event_type", "log"),
        }

        request_id = request_id_var.get()
        if request_id:
            payload["requestId"] = request_id

        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(_sanitize(fields))

        message = record.getMessage()
        if message:
            payload.setdefault("msg", message[:_MAX_STRING])

        if record.exc_info and record.exc_info[0] is not None:
            payload["errorName"] = record.exc_info[0].__name__
            payload["errorMessage"] = str(record.exc_info[1])[:_MAX_STRING]
            # 전체 스택은 한 줄 로그를 수 KB로 만든다. 원인은 대개 끝쪽 몇 프레임에 있다.
            payload["stack"] = "".join(traceback.format_exception(*record.exc_info))[-_MAX_STACK:]

        # 로거 이름이 uvicorn·라이브러리인 경우 출처를 남긴다(직접 낸 이벤트에는 불필요).
        if record.name not in ("standin", "root"):
            payload.setdefault("logger", record.name)

        return json.dumps(payload, ensure_ascii=False, default=str)


logger = logging.getLogger("standin")


def configure_logging() -> None:
    """루트 로거와 uvicorn 로거를 JSON 포맷으로 통일한다. 여러 번 불러도 안전하다."""
    global _configured
    if _configured:
        return
    _configured = True

    # 로그에 한글이 들어간다. stdout 인코딩이 UTF-8이 아니면(윈도우 콘솔 등)
    # 로깅이 UnicodeEncodeError로 죽거나 글자가 깨진다 — 로깅이 장애 원인이 되면 안 된다.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, CFG.log_level.upper(), logging.INFO))

    # uvicorn은 자체 핸들러로 사람이 읽는 형식을 낸다. 그대로 두면 로그가 두 종류로 섞인다.
    for name in ("uvicorn", "uvicorn.error"):
        target = logging.getLogger(name)
        target.handlers = []
        target.propagate = True

    # 접근 로그는 우리 미들웨어가 http_request로 낸다(라우트 패턴·requestId 포함).
    # uvicorn의 접근 로그를 함께 켜면 같은 요청이 두 줄로 남는다.
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False


def log_event(level: int, event_type: str, msg: str = "", exc_info: bool = False, **fields) -> None:
    """공통 스키마 이벤트 한 줄.

    `event_type`은 집계·알림의 1차 키다. 새 값을 만들면 문서 §4 목록에 추가한다.
    """
    logger.log(level, msg, exc_info=exc_info, extra={"event_type": event_type, "fields": fields})


def log_info(event_type: str, msg: str = "", **fields) -> None:
    log_event(logging.INFO, event_type, msg, **fields)


def log_warn(event_type: str, msg: str = "", **fields) -> None:
    log_event(logging.WARNING, event_type, msg, **fields)


def log_error(event_type: str, msg: str = "", exc_info: bool = False, **fields) -> None:
    log_event(logging.ERROR, event_type, msg, exc_info=exc_info, **fields)
