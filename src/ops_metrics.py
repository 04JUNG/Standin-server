"""요청 지표를 분 단위로 모은다(계획 3단계).

추론 서버는 DB가 없고 내부 전용이라 스스로 저장하지 못한다. 대신 최근 몇 분치를
메모리에 들고 `GET /ops/metrics`로 내보내고, **BFF가 1분마다 긁어 가** RDS에 넣는다.

버킷을 내보낸 뒤 지우지 않는 이유: BFF가 쓰다 실패하면 그 분이 통째로 사라진다.
BFF 쪽 저장이 (분, 서비스, 태스크) 기본키 upsert라 같은 값을 다시 읽어도 결과가
같으므로, 여기서는 시간 상한(기본 15분)으로만 오래된 것을 버린다.

지연시간을 p50/p95 값이 아니라 **히스토그램**으로 내보낸다. 태스크별 p95를 나중에
평균 내는 것은 통계적으로 의미가 없지만(p95의 평균은 p95가 아니다), 버킷 카운트는
더할 수 있다. BFF의 src/ops/metrics.ts와 경계값이 같아야 합산이 성립한다.
"""
from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone

# BFF의 LATENCY_BUCKETS_MS와 **반드시 같아야 한다**. 한쪽만 바꾸면 합산이 어긋난다.
LATENCY_BUCKETS_MS = (50, 100, 250, 500, 1000, 2500, 5000, 10_000, 30_000)

# 라우트·에러코드는 원래 유한하지만 버그나 공격으로 값이 폭발할 수 있다.
_MAX_KEYS = 50
_OVERFLOW_KEY = "_other"

# 이 프로세스 식별자. 태스크마다 다른 행을 쓰기 위한 값이라 무작위면 충분하다.
TASK_ID = f"{os.getpid()}-{uuid.uuid4().hex[:6]}"


def _bucket_key(timestamp: float) -> str:
    minute = int(timestamp // 60) * 60
    return datetime.fromtimestamp(minute, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _latency_index(duration_ms: float) -> int:
    for index, upper in enumerate(LATENCY_BUCKETS_MS):
        if duration_ms <= upper:
            return index
    return len(LATENCY_BUCKETS_MS)


def _bump(counter: dict, key: str) -> None:
    if key not in counter and len(counter) >= _MAX_KEYS:
        counter[_OVERFLOW_KEY] = counter.get(_OVERFLOW_KEY, 0) + 1
        return
    counter[key] = counter.get(key, 0) + 1


class MetricsCollector:
    """분 버킷 수집기. 시간을 인자로 받으므로 타이머 없이 검증할 수 있다."""

    def __init__(self, max_buckets: int = 15) -> None:
        self._buckets: dict[str, dict] = {}
        self._max_buckets = max_buckets
        self._lock = threading.Lock()

    def record(self, now: float, status: int, duration_ms: float,
               route: str | None = None, error_code: str | None = None) -> None:
        key = _bucket_key(now)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = {
                    "bucketAt": key,
                    "requests": 0,
                    "errors4xx": 0,
                    "errors5xx": 0,
                    "durationSumMs": 0,
                    "latency": [0] * (len(LATENCY_BUCKETS_MS) + 1),
                    "byError": {},
                    "byRoute": {},
                }
                self._buckets[key] = bucket
                self._evict()

            bucket["requests"] += 1
            bucket["durationSumMs"] += max(0, round(duration_ms))
            bucket["latency"][_latency_index(duration_ms)] += 1
            if status >= 500:
                bucket["errors5xx"] += 1
            elif status >= 400:
                bucket["errors4xx"] += 1
            if error_code:
                _bump(bucket["byError"], error_code)
            if route:
                _bump(bucket["byRoute"], route)

    def snapshot(self) -> list[dict]:
        """지금까지 모인 버킷 전부. 지우지 않는다 — 수집자가 실패해도 다음에 다시 읽는다."""
        with self._lock:
            return [dict(bucket, latency=list(bucket["latency"]),
                         byError=dict(bucket["byError"]), byRoute=dict(bucket["byRoute"]))
                    for bucket in sorted(self._buckets.values(), key=lambda b: b["bucketAt"])]

    def _evict(self) -> None:
        while len(self._buckets) > self._max_buckets:
            oldest = min(self._buckets)
            del self._buckets[oldest]


COLLECTOR = MetricsCollector()
