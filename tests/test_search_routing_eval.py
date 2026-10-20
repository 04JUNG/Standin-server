"""Evaluation must distinguish replay evidence from live Gemini measurements."""
from types import SimpleNamespace

from scripts.eval_search_routing import _write_report, main


def test_report_counts_latency_per_cut_and_labels_replay(tmp_path):
    cases = [
        {"routing_summary": {"describe": {
            "requested_people": 2, "status": "ok", "elapsed_ms": 1000}}},
        {"routing_summary": {"describe": {
            "requested_people": 1, "status": "error", "elapsed_ms": 3000}}},
    ]
    table = [
        {"person": 0, "route": "geometric", "search_status": "ok",
         "same_top5": False, "outcome": "test", "semantic_status": "not_requested"},
        {"person": 1, "route": "semantic", "search_status": "ok",
         "same_top5": False, "outcome": "test", "semantic_status": "valid"},
    ]
    args = SimpleNamespace(out=tmp_path, vlm="live")
    _write_report(args, cases, table)
    report = (tmp_path / "REPORT.md").read_text()
    assert "회귀: 1/1명" in report
    assert "p50=2000.0ms, p95=2900.0ms" in report
    args.vlm = "replay"
    _write_report(args, cases, table)
    report = (tmp_path / "REPORT.md").read_text()
    assert "p50=" not in report and "실제 호출 지연은 미측정" in report


def test_live_evaluation_rejects_mock_fallback(monkeypatch, tmp_path):
    import pytest
    from scripts import eval_search_routing as evaluation
    from src.vlm.client import MockVLMClient

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("test_id,test_path\nT001,input.png\n")
    monkeypatch.setattr(evaluation, "load_entries", lambda path: [])
    monkeypatch.setattr(evaluation, "build_vlm_client", lambda: MockVLMClient())
    with pytest.raises(RuntimeError, match="mock fallback rejected"):
        main(["--manifest", str(manifest), "--images", str(tmp_path),
              "--db", "unused.db", "--tags", "unused.json",
              "--vlm", "live", "--out", str(tmp_path / "output")])
