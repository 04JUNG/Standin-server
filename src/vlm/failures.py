"""Safe, SDK-independent classification; never expose raw provider errors."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def failure_reason(status: int | None, *, timed_out: bool = False) -> str:
    if timed_out:
        return "timeout"
    if status == 429:
        return "rate_limited"
    if status == 503:
        return "upstream_unavailable"
    return "upstream_error"


def _seconds(value) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def classify_failure(error: Exception, status: int | None, *, timed_out=False):
    """Return (reason, minimum retry delay). Unknown 429 is NOT daily quota.

    google-genai APIError.details contains the complete response JSON. Only
    explicit QuotaFailure per-day metadata establishes daily quota exhaustion.
    RESOURCE_EXHAUSTED or free-tier wording alone is not enough evidence.
    """
    reason = failure_reason(status, timed_out=timed_out)
    payload = getattr(error, "details", None)
    if isinstance(payload, dict):
        payload = payload.get("error", payload)
    details = payload.get("details", []) if isinstance(payload, dict) else []
    if not isinstance(details, list):
        details = []
    delays = []
    for item in details:
        if not isinstance(item, dict):
            continue
        if str(item.get("@type", "")).endswith("google.rpc.QuotaFailure") and status == 429:
            violations = item.get("violations", [])
            if not isinstance(violations, list):
                continue
            for violation in violations:
                if not isinstance(violation, dict):
                    continue
                quota = " ".join(str(violation.get(k, "")) for k in ("quotaId", "quotaMetric"))
                if re.search(r"per[_ -]?day|daily", quota, re.IGNORECASE):
                    reason = "daily_quota_exhausted"
        if str(item.get("@type", "")).endswith("google.rpc.RetryInfo"):
            duration = item.get("retryDelay")
            seconds = None
            if isinstance(duration, str) and re.fullmatch(r"\d+(?:\.\d+)?s", duration):
                seconds = _seconds(duration[:-1])
            if seconds is not None:
                delays.append(seconds)
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw = headers.get("Retry-After")
        seconds = _seconds(raw)
        if seconds is None and isinstance(raw, str):
            try:
                date = parsedate_to_datetime(raw)
                seconds = max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
        if seconds is not None:
            delays.append(seconds)
    # Multiple hints must never shorten the upstream's requested cooldown.
    return reason, max(delays) if delays else None
