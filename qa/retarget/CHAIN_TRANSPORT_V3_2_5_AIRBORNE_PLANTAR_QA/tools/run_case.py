"""Run one V3.2.5 airborne plantar QA conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airborne_plantar_safe import convert_safe, math_self_test  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--force-exact-v324", action="store_true")
    parser.add_argument(
        "--output-mode", default="rigged_rest",
        choices=("rigged_rest", "static_mesh"),
    )
    args = parser.parse_args(argv)
    print("[SELF_TEST] " + json.dumps(math_self_test(), sort_keys=True))
    report = convert_safe(
        bvh_path=args.bvh,
        character_fbx=args.character,
        out_path=args.out,
        frame=args.frame,
        mirror=args.mirror,
        output_mode=args.output_mode,
        force_exact_v324=args.force_exact_v324,
    )
    payload = json.dumps(report.as_dict(), ensure_ascii=False, indent=2)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.ok else 1


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(argv))
