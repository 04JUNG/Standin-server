from __future__ import annotations

from scripts.measure_converter_latency import (
    _latency_summary,
    _multipart_body,
    _percentile_nearest_rank,
)


def test_nearest_rank_percentile_is_deterministic():
    values = [20.0, 1.0, 10.0, 30.0, 40.0]
    assert _percentile_nearest_rank(values, 0.50) == 20.0
    assert _percentile_nearest_rank(values, 0.95) == 40.0


def test_latency_summary_keeps_queue_and_execution_separate():
    samples = [
        {"queue_ms": 1.0, "execution_ms": 10.0},
        {"queue_ms": 2.0, "execution_ms": 20.0},
        {"queue_ms": 3.0, "execution_ms": 30.0},
    ]
    assert _latency_summary(samples, "queue_ms") == {
        "p50_ms": 2.0,
        "p95_ms": 3.0,
        "max_ms": 3.0,
    }
    assert _latency_summary(samples, "execution_ms")["p95_ms"] == 30.0


def test_latency_probe_builds_locked_converter_request():
    body, content_type = _multipart_body(
        bvh=b"HIERARCHY\nMOTION\n",
        filename="pose.bvh",
        character_id="standin-master-v2",
    )
    assert content_type.startswith("multipart/form-data; boundary=standin-")
    assert b'name="character_id"\r\n\r\nstandin-master-v2' in body
    assert b'name="frame"\r\n\r\n0' in body
    assert b'name="mirror"\r\n\r\nfalse' in body
    assert b'name="output_mode"\r\n\r\nrigged_rest' in body
    assert b'name="apply_root_translation"\r\n\r\nfalse' in body
    assert b'filename="pose.bvh"' in body
    assert b"HIERARCHY\nMOTION\n" in body
