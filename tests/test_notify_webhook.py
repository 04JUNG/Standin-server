"""디스코드 웹훅 전송이 Cloudflare에 막히지 않는지 지킨다.

2026-08-19 프로덕션에서 추론 서버 알림이 **한 건도 배달되지 않았다**(전송 3건 전부 403).
원인은 웹훅 URL도 시크릿도 아니고 요청 헤더였다 — urllib이 기본으로 붙이는
`Python-urllib/x.y`를 디스코드 앞단 Cloudflare가 403(error code: 1010)으로 막는다.
BFF(Node fetch)는 기본 UA가 달라 멀쩡했던 탓에 한쪽 서비스에서만 조용히 깨져 있었다.

`_send`는 서비스를 죽이지 않으려고 전송 예외를 전부 삼킨다(그게 맞다). 그래서 이 경로는
테스트가 없으면 다음에도 조용히 깨진다. 여기서는 "무엇을 보내는가"가 아니라
"보낼 수 있는 모양으로 보내는가"만 고정한다.
"""
import json
import urllib.request

import pytest

from src.config import CFG
from src import notify as notify_mod


@pytest.fixture
def captured_request(monkeypatch):
    """웹훅을 설정하고 실제 전송을 가로채 Request 객체를 돌려준다."""
    monkeypatch.setattr(CFG, "discord_webhook_alert", "https://discord.test/webhook-alert")
    monkeypatch.setattr(CFG, "discord_webhook_warn", "https://discord.test/webhook-warn")
    monkeypatch.setattr(CFG, "discord_webhook_ops", "https://discord.test/webhook-ops")
    captured = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, **_kwargs: captured.append(request) or _Response()
    )
    return captured


def _entry(severity="P2"):
    return notify_mod._Pending(
        severity=severity, code="INFERENCE_FAILED", message="분석 실패", last_at=0.0
    )


def test_sets_explicit_user_agent(captured_request):
    """UA를 명시하지 않으면 urllib 기본값이 붙고, 디스코드가 403으로 막는다."""
    notify_mod._send("P2", [_entry()])

    assert len(captured_request) == 1
    sent = captured_request[0]
    user_agent = sent.get_header("User-agent")
    assert user_agent, "User-Agent를 명시하지 않으면 Cloudflare가 403(1010)으로 막는다"
    assert "urllib" not in user_agent.lower(), f"urllib 기본 UA는 차단된다: {user_agent}"


def test_keeps_json_content_type(captured_request):
    """UA를 넣다가 Content-Type을 잃으면 디스코드가 본문을 파싱하지 못한다."""
    notify_mod._send("P2", [_entry()])

    sent = captured_request[0]
    assert sent.get_header("Content-type") == "application/json"
    assert json.loads(sent.data.decode("utf-8"))["embeds"]


@pytest.mark.parametrize("severity", ["P1", "P2", "P3"])
def test_all_severities_go_out_with_user_agent(captured_request, severity):
    """P1도 같은 전송 경로를 탄다 — 등급별로 빠지는 곳이 없어야 한다."""
    notify_mod._send(severity, [_entry(severity)])

    user_agent = captured_request[0].get_header("User-agent")
    assert user_agent, f"{severity} 전송에 User-Agent가 빠졌다"
    assert "urllib" not in user_agent.lower()


def test_no_webhook_configured_sends_nothing(monkeypatch, captured_request):
    """로컬처럼 웹훅이 없는 환경에서는 전송 자체를 하지 않는다(기존 동작 유지)."""
    monkeypatch.setattr(CFG, "discord_webhook_alert", "")
    monkeypatch.setattr(CFG, "discord_webhook_warn", "")
    monkeypatch.setattr(CFG, "discord_webhook_ops", "")

    notify_mod._send("P2", [_entry()])

    assert captured_request == []
