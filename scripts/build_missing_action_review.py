"""Build multi-view sheets and a CSV for missing action-name review."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("front", "three_quarter", "side", "back")


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_proposals(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["semantic_unit_id"]: row for row in data.get("proposals", [])}


def build_rows(
    missing_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
    proposals: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    source_numbers = {row["source_clip_id"]: row["no"] for row in missing_rows}
    output = []
    for row in review_rows:
        if row.get("action_name_status") != "missing":
            continue
        member_ids = json.loads(row["member_pose_ids"])
        pose_id = next((item for item in member_ids if not item.endswith("_mirror")), member_ids[0])
        proposal = proposals.get(row["semantic_unit_id"], {})
        output.append(
            {
                "no": source_numbers[row["source_clip_id"]],
                "source_provider": row["source_provider"],
                "source_clip_id": row["source_clip_id"],
                "semantic_unit_id": row["semantic_unit_id"],
                "pose_id": pose_id,
                "pose_description": row["caption_ko_proposed"],
                "action_name_proposed_ko": proposal.get("action_name_proposed_ko", ""),
                "action_name_proposed_en": proposal.get("action_name_proposed_en", ""),
                "confidence": proposal.get("confidence", ""),
                "visual_evidence": proposal.get("visual_evidence", ""),
                "ambiguity": proposal.get("ambiguity", ""),
                "decision": "",
                "action_name_final": "",
                "reviewer_notes": "",
            }
        )
    return sorted(output, key=lambda row: (int(row["no"]), row["semantic_unit_id"]))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_sheets(
    output_dir: Path,
    thumb_dir: Path,
    rows: list[dict[str, str]],
    rows_per_sheet: int,
    sheet_title: str,
) -> None:
    cell = 256
    label_width = 520
    header_height = 50
    row_height = 286
    title_font = _font(20)
    label_font = _font(16)
    small_font = _font(13)
    output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), rows_per_sheet):
        page_rows = rows[start : start + rows_per_sheet]
        sheet = Image.new(
            "RGB",
            (label_width + cell * len(VIEWS), header_height + row_height * len(page_rows)),
            "#f6f2e9",
        )
        draw = ImageDraw.Draw(sheet)
        page_number = start // rows_per_sheet + 1
        draw.text((16, 12), f"{sheet_title} - {page_number}", fill="#222222", font=title_font)
        for view_index, view in enumerate(VIEWS):
            draw.text(
                (label_width + view_index * cell + 10, 18), view,
                fill="#555555", font=small_font,
            )
        for row_index, row in enumerate(page_rows):
            top = header_height + row_index * row_height
            draw.rectangle((0, top, sheet.width, top + row_height - 1), outline="#c8bea9", width=1)
            label_lines = [
                f"#{row['no']}  {row['source_clip_id']}",
                row["pose_id"],
            ]
            if row["action_name_proposed_ko"]:
                label_lines.append(f"제안: {row['action_name_proposed_ko']}")
                label_lines.append(f"신뢰도: {row['confidence']}")
                label_lines.append("애매한 이유:")
                label_lines.extend(textwrap.wrap(row["ambiguity"], width=28))
            draw.multiline_text((14, top + 24), "\n".join(label_lines), fill="#222222", font=label_font, spacing=8)
            for view_index, view in enumerate(VIEWS):
                path = thumb_dir / f"{row['pose_id']}__{view}.png"
                left = label_width + view_index * cell
                if path.is_file():
                    image = Image.open(path).convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
                    sheet.paste(image, (left, top + 15))
                else:
                    draw.rectangle((left, top + 15, left + cell - 1, top + 15 + cell - 1), fill="#dddddd")
                    draw.text((left + 20, top + 120), "missing thumbnail", fill="#aa3333", font=small_font)
        sheet.save(output_dir / f"review_sheet_{page_number:02d}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-csv", default="data/semantic/missing_action_names.csv")
    parser.add_argument("--review-csv", default="data/semantic/review_queue.csv")
    parser.add_argument("--thumb-dir", default="data/thumbs")
    parser.add_argument("--proposals-json")
    parser.add_argument("--output-dir", default="data/semantic/action_name_review")
    parser.add_argument("--rows-per-sheet", type=int, default=6)
    parser.add_argument("--confidence", nargs="+", choices=("high", "medium", "low"))
    parser.add_argument("--sheet-title", default="행동명 검수")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = build_rows(
        _read_csv(Path(args.missing_csv)),
        _read_csv(Path(args.review_csv)),
        _load_proposals(Path(args.proposals_json) if args.proposals_json else None),
    )
    if args.confidence:
        rows = [row for row in rows if row["confidence"] in args.confidence]
    if not rows:
        raise ValueError("no missing action rows found")
    write_csv(output_dir / "action_name_review.csv", rows)
    write_sheets(output_dir, Path(args.thumb_dir), rows, args.rows_per_sheet, args.sheet_title)
    print(json.dumps({"rows": len(rows), "sheets": (len(rows) + args.rows_per_sheet - 1) // args.rows_per_sheet}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
