"""Refresh the existing body asset initializer to the immutable serving image."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

IMAGE = re.compile(r"[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/standin/inference:[0-9a-f]{40}\Z")


def prepare(task, image):
    if not IMAGE.fullmatch(image):
        raise ValueError("immutable inference commit image required")
    result = copy.deepcopy(task)
    containers = result.get("containerDefinitions", [])
    initializers = [c for c in containers if c.get("name") == "body-assets-init"]
    if not initializers:
        return result
    if len(initializers) != 1 or len([c for c in containers if c.get("name") == "inference"]) != 1:
        raise ValueError("ambiguous inference containers")
    initializer = initializers[0]
    command = initializer.get("command", [])
    if command[:2] != ["python", "-c"] or len(command) != 3 or "body_assets_ready" not in command[2]:
        raise ValueError("unknown body initializer contract")
    initializer["image"] = image
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    result = prepare(json.loads(args.task.read_text()), args.image)
    args.task.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
