"""장애 알림(디스코드 웹훅). 설계 정본:
Standin-master-docs/관측성_로그모니터링_알림_2026-08-18.md §5.

이 모듈의 값어치는 "무엇을 보내는가"가 아니라 **"어떻게 안 보내는가"**에 있다.
디스코드 웹훅에는 자체 레이트리밋이 있어서, 장애 때 초당 수백 건의 에러를 그대로 쏘면
레이트리밋에 걸려 알림이 통째로 유실된다. 그래서 세 겹으로 줄인다.
  1. 배치      — flush 창(기본 10초) 안의 이벤트를 한 메시지로 묶는다.
  2. 중복 억제 — 같은 키는 억제 창(기본 5분) 동안 첫 건만 보내고 나머지는 세기만 한다.
  3. 상한      — 한 메시지의 임베드를 5개로 자르고 나머지는 "외 N종"으로 요약한다.

의존성을 늘리지 않으려고 표준 라이브러리(urllib)만 쓴다. 전송은 데몬 스레드에서 하므로
FastAPI 요청 스레드를 붙잡지 않는다.

⚠ 알림 실패가 서비스를 죽이면 안 된다. 전송 오류는 전부 삼키고 로그만 남긴다.
  그 로그는 절대 notify()를 다시 부르지 않는다(무한 재귀).
"""
from __future__ import annotations

import atexit
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import CFG
from .logging_setup import log_warn, request_id_var

# P1 — 사람을 깨운다. 서비스가 죽었거나 잘못된 결과를 서빙 중이다.
# P2 — 업무시간에 본다. 일부 요청이 실패하거나 품질이 떨어진다.
# P3 — 기록. 기동·배포·일일 요약.
SEVERITIES = ("P1", "P2", "P3")

_COLORS = {"P1": 0xE03131, "P2": 0xF08C00, "P3": 0x868E96}
_ICONS = {"P1": "🔴", "P2": "🟠", "P3": "⚪"}

# 디스코드 임베드 필드 값 상한은 1024자다. 여유를 두고 자른다.
_MAX_FIELD_VALUE = 900
_SEND_TIMEOUT_SECONDS = 5


@dataclass
class _Pending:
    severity: str
    code: str
    message: str
    context: dict = field(default_factory=dict)
    count: int = 1
    last_at: float = 0.0
    request_id: str | None = None


_lock = threading.Lock()
_pending: dict[str, _Pending] = {}
# key → 억제 해제 시각. 이 시각 전의 재발은 보내지 않고 아래 카운터만 올린다.
_suppressed_until: dict[str, float] = {}
# 억제 창 동안 삼킨 횟수. 창이 끝나고 처음 재발할 때 "×N"으로 함께 보고한다.
_suppressed_counts: dict[str, int] = {}
_worker: threading.Thread | None = None
_wake = threading.Event()


def _webhook_for(severity: str) -> str:
    """채널을 하나만 만든 팀도 그대로 동작해야 한다 — 없으면 다른 채널로 흘린다."""
    alert, warn, ops = (
        CFG.discord_webhook_alert,
        CFG.discord_webhook_warn,
        CFG.discord_webhook_ops,
    )
    if severity == "P1":
        return alert or warn or ops
    if severity == "P2":
        return warn or alert or ops
    return ops or warn or alert


def notify(
    severity: str,
    code: str,
    message: str,
    key: str | None = None,
    context: dict | None = None,
) -> None:
    """알림을 큐에 넣는다. 절대 예외를 던지지 않는다.

    웹훅이 설정되지 않은 환경(로컬)에서는 큐에 넣되 전송하지 않는다. 같은 사건이 이미
    로그에 남아 있으므로 개발 중에 놓치는 정보가 없다.
    """
    dedupe_key = key or f"{severity}:{code}"
    now = time.time()

    with _lock:
        existing = _pending.get(dedupe_key)
        if existing is not None:
            existing.count += 1
            existing.last_at = now
            return

        until = _suppressed_until.get(dedupe_key)
        if until is not None and now < until:
            _suppressed_counts[dedupe_key] = _suppressed_counts.get(dedupe_key, 0) + 1
            return

        carried = _suppressed_counts.pop(dedupe_key, 0)
        _suppressed_until.pop(dedupe_key, None)
        _pending[dedupe_key] = _Pending(
            severity=severity if severity in SEVERITIES else "P2",
            code=code,
            message=message,
            context=dict(context or {}),
            count=1 + carried,
            last_at=now,
            request_id=request_id_var.get(),
        )

    _ensure_worker()


def notify_now(severity: str, code: str, message: str, context: dict | None = None) -> None:
    """큐에 넣고 즉시 보낸다. 곧 프로세스가 종료되는 경로에서만 쓴다(기동 실패 등)."""
    notify(severity, code, message, context=context)
    flush()


def flush() -> None:
    """큐를 즉시 비운다. 다음 배치를 기다릴 수 없을 때 쓴다."""
    with _lock:
        if not _pending:
            return
        entries = list(_pending.items())
        _pending.clear()
        deadline = time.time() + CFG.alert_suppress_seconds
        for dedupe_key, _ in entries:
            _suppressed_until[dedupe_key] = deadline

    grouped = [entry for _, entry in entries]
    for severity in SEVERITIES:
        group = [entry for entry in grouped if entry.severity == severity]
        if group:
            _send(severity, group)


def _ensure_worker() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        _wake.set()
        return
    _worker = threading.Thread(target=_loop, name="standin-notify", daemon=True)
    _worker.start()


def _loop() -> None:
    while True:
        # 배치 창만큼 기다렸다가 비운다. 대기 중 새 알림이 와도 창을 앞당기지 않는다
        # (앞당기면 배치의 목적인 "묶기"가 무너진다).
        _wake.wait(CFG.alert_flush_seconds)
        _wake.clear()
        try:
            flush()
        except Exception:  # noqa: BLE001 — 알림 스레드는 어떤 이유로도 죽지 않는다.
            log_warn("notify_failed", errorCode="FLUSH_FAILED")


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "…"


def _describe(entry: _Pending) -> str:
    lines = [_truncate(entry.message, _MAX_FIELD_VALUE)]
    if entry.count > 1:
        lines.append(f"발생 {entry.count}건")
    if entry.request_id:
        lines.append(f"requestId: {entry.request_id}")
    for key, value in entry.context.items():
        lines.append(f"{key}: {_truncate(str(value), 200)}")
    return _truncate("\n".join(lines), _MAX_FIELD_VALUE)


def _send(severity: str, entries: list[_Pending]) -> None:
    webhook = _webhook_for(severity)
    if not webhook:
        return  # 미설정(로컬) — 로그에는 이미 같은 사건이 남아 있다.

    shown = entries[: CFG.alert_max_per_flush]
    hidden = entries[CFG.alert_max_per_flush :]
    footer = {"text": f"standin/inference · {CFG.app_env} · {CFG.deployment_version}"}

    embeds = [
        {
            "title": _truncate(f"{_ICONS[severity]} {severity} · {entry.code}", 256),
            "description": _describe(entry),
            "color": _COLORS[severity],
            "timestamp": datetime.fromtimestamp(entry.last_at, tz=timezone.utc).isoformat(),
            "footer": footer,
        }
        for entry in shown
    ]
    if hidden:
        embeds.append(
            {
                "title": f"… 외 {len(hidden)}종",
                "description": _truncate(", ".join(e.code for e in hidden), _MAX_FIELD_VALUE),
                "color": _COLORS[severity],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": footer,
            }
        )

    mention = CFG.discord_alert_mention if severity == "P1" else ""
    body: dict = {
        "embeds": embeds,
        # 멘션을 명시적으로 켜지 않으면 디스코드가 @here를 실제 알림으로 처리하지 않는다.
        "allowed_mentions": {"parse": ["everyone", "roles"]} if mention else {"parse": []},
    }
    if mention:
        body["content"] = mention

    request = urllib.request.Request(
        webhook,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_SEND_TIMEOUT_SECONDS):
            pass
    except urllib.error.HTTPError as error:
        # 여기서 notify()를 부르면 실패가 실패를 부른다. 로그만 남긴다.
        log_warn("notify_failed", severity=severity, status=error.code, errorCode="WEBHOOK_REJECTED")
    except Exception as error:  # noqa: BLE001
        log_warn(
            "notify_failed",
            severity=severity,
            errorCode="WEBHOOK_UNREACHABLE",
            errorName=type(error).__name__,
        )


# 종료 직전에 버퍼가 사라지면 마지막 알림을 놓친다.
atexit.register(flush)
