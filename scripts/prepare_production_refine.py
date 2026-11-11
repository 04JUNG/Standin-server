"""Pin the approved production refine path without enabling staging experiments."""
import json
import sys
from pathlib import Path

POLICY = {"REFINE_V2_ENABLED": "1",
          "REFINE_DEFAULT_MODE": "aggressive", "REFINE_V25_SELECTOR_ENABLED": "1",
          "REFINE_V2_TORSO": "0", "REFINE_V3_MODE": "off",
          "REFINE_V3_EVIDENCE_ENABLED": "0", "REFINE_RELEASE_ENABLED": "0"}

def prepare(task):
    containers = [c for c in task["containerDefinitions"] if c["name"] == "inference"]
    if len(containers) != 1:
        raise ValueError("exactly one inference container required")
    container = containers[0]
    env = {v["name"]: v["value"] for v in container.get("environment", [])}
    # Preserve the global emergency kill switch while pinning the approved engine.
    env.setdefault("REFINE_ENABLED", "1")
    if env["REFINE_ENABLED"] not in ("0", "1"):
        raise ValueError("invalid production refine kill switch")
    env.update(POLICY)
    env.pop("REFINE_RELEASE_WORKER_URL", None)
    container["environment"] = [{"name": k, "value": v} for k, v in env.items()]
    return task

if __name__ == "__main__":
    path = Path(sys.argv[1])
    path.write_text(json.dumps(prepare(json.loads(path.read_text())), indent=2) + "\n")
