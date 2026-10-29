"""Apply explicit description rollout overrides to a downloaded ECS task.

Empty GitHub environment variables preserve existing task settings. No AWS calls,
secret lookup, image changes, or implicit enablement of production flags.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


FLAGS = {"PARALLEL_POSE_DESCRIPTION", "ROUTING_DESCRIBE_PERSON_CROPS"}
TIMINGS = {"ROUTING_DESCRIBE_TIMEOUT_MS", "ROUTING_DESCRIBE_BUDGET_MS",
           "ROUTING_DESCRIBE_RESERVE_MS"}
WORKERS = "ROUTING_DESCRIBE_PERSON_WORKERS"
NAMES = FLAGS | TIMINGS | {WORKERS}


def prepare(task, overrides):
    values = {}
    for name, value in overrides.items():
        if name not in NAMES:
            raise ValueError("unsupported description setting")
        if value in (None, ""):
            continue
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
            raise ValueError(f"invalid {name}")
        if name in FLAGS and value not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
        if name == WORKERS and value not in ("1", "2"):
            raise ValueError(f"{name} must be 1 or 2")
        if name in TIMINGS and (int(value) < (0 if name.endswith("RESERVE_MS") else 1)
                                or int(value) > 120000):
            raise ValueError(f"{name} must fit within the 120s request budget")
        values[name] = value
    result = copy.deepcopy(task)
    containers = [c for c in result["containerDefinitions"] if c["name"] == "inference"]
    if len(containers) != 1:
        raise ValueError("exactly one inference container required")
    container = containers[0]
    rows = container.get("environment", [])
    env = {r["name"]: r["value"] for r in rows}
    if len(env) != len(rows):
        raise ValueError("duplicate environment names in task definition")
    if values.keys() & {s["name"] for s in container.get("secrets", [])}:
        raise ValueError("description setting conflicts with an existing secret")
    if values:
        env.update(values)
        container["environment"] = [{"name": k, "value": v} for k, v in env.items()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    args = parser.parse_args()
    overrides = {name: os.getenv(name, "") for name in sorted(NAMES)}
    result = prepare(json.loads(args.task.read_text()), overrides)
    args.task.write_text(json.dumps(result), encoding="utf-8")
    print("description rollout settings prepared; unspecified settings preserved")


if __name__ == "__main__":
    main()
