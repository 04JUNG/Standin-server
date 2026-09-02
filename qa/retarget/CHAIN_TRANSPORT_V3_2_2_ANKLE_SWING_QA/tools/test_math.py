"""Blender entry point for the V3.2.2 pure geometry controls."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ankle_swing_safe import math_self_test  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(math_self_test(), indent=2, sort_keys=True))
