"""V3.2.1 Palm Roll QA 단일 변환. 운영 converter는 import하지 않는다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.convert import convert  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--output-mode", default="rigged_rest")
    parser.add_argument(
        "--palm-roll-mu", type=float, default=None,
        choices=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="omit to use palm_roll_policy.json default (currently 0.5)",
    )
    parser.add_argument("--palm-roll-mu-left", type=float)
    parser.add_argument("--palm-roll-mu-right", type=float)
    args = parser.parse_args(argv)
    selected_mu = args.palm_roll_mu
    if args.palm_roll_mu_left is not None or args.palm_roll_mu_right is not None:
        if args.palm_roll_mu_left is None or args.palm_roll_mu_right is None:
            raise ValueError("left/right palm mu must be supplied together")
        selected_mu = {
            "hand.L": args.palm_roll_mu_left,
            "hand.R": args.palm_roll_mu_right,
        }
    report = convert(
        bvh_path=args.bvh,
        character_fbx=args.character,
        out_path=args.out,
        frame=args.frame,
        mirror=args.mirror,
        output_mode=args.output_mode,
        palm_roll_mu=selected_mu,
    )
    payload = report.as_dict()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PALM_QA_IMPORT_ROOT=" + str(ROOT))
    print("PALM_QA_REPORT=" + json.dumps(payload, ensure_ascii=False))
    return 0 if report.ok else 2


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(argv))
