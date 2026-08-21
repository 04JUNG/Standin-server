"""실제 google-genai HTTP transport를 사용해 Gemini 장애 경계를 검증한다.

외부 Gemini나 API 키에는 의존하지 않는다. 로컬 HTTP 서버가 무응답/503/정상 응답을
결정적으로 재현하므로 CI에서도 timeout과 제한 재시도가 실제 소켓 수준에서 동작한다.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from google import genai
from google.genai import types

from src.config import CFG
from src.vlm.client import GeminiVLMClient, VLMUnavailable


VALID_ANALYSIS = {
    "num_people": 1,
    "shot": "full_half",
    "action": "standing",
    "view": "front",
    "relationship": "solo",
    "approx_boxes": [],
}


def gemini_response() -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(VALID_ANALYSIS)}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "modelVersion": "fault-test",
    }


@contextmanager
def fault_server(outcomes):
    state = {"outcomes": list(outcomes), "requests": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler contract
            index = state["requests"]
            state["requests"] += 1
            outcome = state["outcomes"][min(index, len(state["outcomes"]) - 1)]
            if outcome == "hang":
                time.sleep(0.5)
                self._json(200, gemini_response())
                return
            if outcome == 503:
                self._json(
                    503,
                    {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}},
                )
                return
            self._json(200, gemini_response())

        def _json(self, status: int, payload: dict):
            body = json.dumps(payload).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # timeout 시 클라이언트가 먼저 소켓을 닫는 것이 기대 동작이다.
                pass

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def client_for(base_url: str, timeout_ms: int, monkeypatch) -> GeminiVLMClient:
    # 요청 단위 timeout은 CFG에서 온다(클라이언트 생성 시 값은 어댑터가 덮어쓴다) —
    # 남은 예산에 맞춰 시도마다 상한을 줄여야 하기 때문이다. 프로덕션도 같은 경로다.
    monkeypatch.setattr(CFG, "gemini_request_timeout_ms", timeout_ms)
    sdk = genai.Client(
        api_key="fault-test-key",
        http_options=types.HttpOptions(base_url=base_url, timeout=timeout_ms),
    )
    client = GeminiVLMClient(
        model="fault-test-model",
        client=sdk,
        sleep=lambda _seconds: None,
        jitter=lambda low, high: (low + high) / 2,
    )
    monkeypatch.setattr(client, "_to_part", lambda _image: "image-part")
    return client


def test_never_responding_transport_hits_real_sdk_timeout(monkeypatch):
    monkeypatch.setattr(CFG, "gemini_max_attempts", 3)
    with fault_server(["hang"]) as (base_url, state):
        client = client_for(base_url, 100, monkeypatch)
        started = time.monotonic()

        with pytest.raises(VLMUnavailable) as raised:
            client.analyze("ignored", 100, 100)

        assert time.monotonic() - started < 1.0
        assert state["requests"] == 1
        # timeout은 재시도하지 않지만, 원인은 상류 지연이므로 503으로 번역될 수 있게
        # VLMUnavailable로 감싼다. 실제 SDK 예외는 원인 사슬에 남는다.
        assert "timeout" in type(raised.value.__cause__).__name__.lower()


def test_real_sdk_503_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(CFG, "gemini_max_attempts", 3)
    with fault_server([503, 503, 200]) as (base_url, state):
        # 이 테스트가 보는 것은 "503을 재시도하는가"다. 부하가 걸린 CI에서 응답이 늦어
        # 데드라인에 걸리면(=timeout, 재시도 대상 아님) 다른 이유로 빨개진다 — 그래서
        # 상한을 넉넉히 준다.
        client = client_for(base_url, 5_000, monkeypatch)

        result = client.analyze("ignored", 100, 100)

        assert result.num_people == 1
        assert state["requests"] == 3
