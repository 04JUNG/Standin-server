"""Selected-target conversion through the pinned approved terminal-bone adapter.

Dependency: PR #170 @ 70b8a6f37d748d5a81801e0c1d632f7c512ba517.
No frozen solver edit, alternate thresholds, or appearance override.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converter.worker import load_job, _write_report
from converter.interactive_adapter import _run_body_job


def run(job):
    import os
    import time
    from converter import retarget as rt
    if os.getenv("STANDIN_LAB_PREVIEW_TRACE") != "1":
        return _run_body_job(job)
    original = rt.export_fbx
    records = []
    def measured(path, **kwargs):
        started = time.monotonic()
        value = original(path, **kwargs)
        records.append({"duration_ms": (time.monotonic()-started)*1000,
                        "bytes": Path(path).stat().st_size})
        return value
    rt.export_fbx = measured
    try:
        report = _run_body_job(job)
    finally:
        rt.export_fbx = original
    report["selected_fbx_serialization"] = records
    return report


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--")+1:]
    if len(args) != 2 or args[0] != "--job":
        raise ValueError("expected server-authored job")
    job = load_job(args[1])
    report = run(job)
    _write_report(Path(job["report_path"]), report)
    raise SystemExit(0 if report.get("ok") is True else 1)
