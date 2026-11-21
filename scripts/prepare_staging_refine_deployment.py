"""Pin staging to legacy v2.5.4 safe-aggressive without altering other containers."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

DEFAULTS = {
    "REFINE_RELEASE_ENABLED": "0",
    "REFINE_ENABLED": "1",
    "REFINE_V2_ENABLED": "1",
    "REFINE_DEFAULT_MODE": "aggressive",
    "REFINE_V25_SELECTOR_ENABLED": "1",
}


def prepare(task, settings=None):
    if not task.get("family", "").startswith("StandinStagingApp"):
        raise ValueError("refine deployment policy is staging-only")
    values = dict(DEFAULTS)
    values.update(settings or {})
    if set(values) != set(DEFAULTS):
        raise ValueError("unexpected refine setting")
    for name, value in values.items():
        allowed = {"conservative", "aggressive"} if name == "REFINE_DEFAULT_MODE" else {"0", "1"}
        if value not in allowed:
            raise ValueError(f"invalid {name}")
    if values["REFINE_DEFAULT_MODE"] == "aggressive" and values["REFINE_V25_SELECTOR_ENABLED"] != "1":
        raise ValueError("aggressive mode requires the safety selector")
    result = copy.deepcopy(task)
    containers = [c for c in result["containerDefinitions"] if c["name"] == "inference"]
    if len(containers) != 1:
        raise ValueError("exactly one inference container required")
    container = containers[0]
    rows = container.get("environment", [])
    env = {r["name"]: r["value"] for r in rows}
    if len(env) != len(rows):
        raise ValueError("duplicate environment setting")
    if values.keys() & {s["name"] for s in container.get("secrets", [])}:
        raise ValueError("refine setting conflicts with a secret")
    env.update(values)
    container["environment"] = [{"name": k, "value": v} for k, v in env.items()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    args = parser.parse_args()
    settings = {name: os.environ.get("STAGING_" + name, value) for name, value in DEFAULTS.items()}
    args.task.write_text(json.dumps(prepare(json.loads(args.task.read_text()), settings)), encoding="utf-8")
    print("Staging refine policy applied; other containers and settings preserved")


if __name__ == "__main__":
    main()
