from types import SimpleNamespace

import httpx
import pytest
from google.genai.errors import ClientError

from src.vlm.failures import classify_failure


def quota_error(quota_id, *, delay=None, header=None):
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": quota_id}]}]
    if delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})
    return ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}},
                       response=httpx.Response(429, headers={"Retry-After": header} if header else {}))


@pytest.mark.parametrize("quota", ["GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                   "generate_content_requests_per_day", "daily_requests"])
def test_only_explicit_daily_quota_is_classified_as_exhausted(quota):
    assert classify_failure(quota_error(quota), 429)[0] == "daily_quota_exhausted"


def test_minute_or_unknown_quota_is_not_daily():
    for quota in ("GenerateRequestsPerMinutePerProjectPerModel", "", "RESOURCE_EXHAUSTED"):
        assert classify_failure(quota_error(quota), 429)[0] == "rate_limited"


def test_retry_hints_take_the_longest_cooldown():
    assert classify_failure(quota_error("per_minute", delay="45.2s", header="60"), 429) == ("rate_limited", 60)


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "bad", "9e999"])
def test_invalid_retry_after_is_ignored(value):
    assert classify_failure(quota_error("", delay=value, header=value), 429)[1] is None


def test_malformed_details_are_safe():
    for details in (None, [], "bad", {"error": None}, {"error": {"details": "bad"}},
                    {"error": {"details": [None, {"@type": "google.rpc.QuotaFailure", "violations": None}]}}):
        assert classify_failure(SimpleNamespace(details=details), 429) == ("rate_limited", None)


def test_http_date_retry_after():
    err = SimpleNamespace(response=httpx.Response(503, headers={"Retry-After": "Wed, 01 Jan 2098 00:00:00 GMT"}))
    assert classify_failure(err, 503)[1] > 60


@pytest.mark.parametrize("status,timeout,expected", [(503, False, "upstream_unavailable"),
    (500, False, "upstream_error"), (None, True, "timeout")])
def test_distinct_other_causes(status, timeout, expected):
    assert classify_failure(Exception("private upstream message"), status, timed_out=timeout)[0] == expected
