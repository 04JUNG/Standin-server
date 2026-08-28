"""로그 마스킹 규칙 검증.

이름 기반 차단(`_REDACT_KEY`)은 새 필드가 자동으로 걸리는 것이 목적이다.
사용량 카운트를 위해 둔 예외가 그 보장을 깨지 않는지 여기서 고정한다.
"""
from src.logging_setup import _sanitize


def test_usage_counts_survive_redaction():
    fields = _sanitize({"promptTokens": 812, "outputTokens": 57, "totalTokens": 869})

    assert fields == {"promptTokens": 812, "outputTokens": 57, "totalTokens": 869}


def test_allowlisted_name_with_non_numeric_value_is_still_redacted():
    """허용 이름에 문자열이 실려 오면(=비밀이 잘못 들어간 경우) 그대로 가린다."""
    fields = _sanitize({"totalTokens": "AIzaSy-secret", "promptTokens": True})

    assert fields["totalTokens"] == "[redacted]"
    assert fields["promptTokens"] == "[redacted]"   # bool은 카운트가 아니다


def test_secret_names_remain_redacted():
    fields = _sanitize({
        "api_key": "AIzaSy-secret",
        "access_token": "abc",
        "authorization": "Bearer x",
        "model": "gemini-2.5-flash",
    })

    assert fields["api_key"] == "[redacted]"
    assert fields["access_token"] == "[redacted]"
    assert fields["authorization"] == "[redacted]"
    assert fields["model"] == "gemini-2.5-flash"
