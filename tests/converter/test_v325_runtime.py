"""Blender executable contract for the promoted V3.2.5 runtime path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from converter.convert import convert


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_path = out_dir / "v325-default.fbx"
    default = convert(
        bvh_path=args.bvh,
        character_fbx=args.character,
        out_path=str(default_path),
        embed_textures=False,
    )
    default_payload = default.as_dict()
    assert default_payload["ok"] is True
    assert default_path.is_file() and default_path.stat().st_size > 0
    selection = default_payload["airborne_plantar_selection"]
    assert selection["selector"] == "V3.2.5_CHAIN_RELATIVE_VIRTUAL_PLANTAR"
    assert selection["status"] == "PROMOTED_RUNTIME"

    source_lines = Path(args.bvh).read_text(encoding="utf-8").splitlines()
    frame_time = next(
        index for index, line in enumerate(source_lines)
        if line.startswith("Frame Time:")
    )
    values = source_lines[frame_time + 1].split()
    values[3] = f"{float(values[3]) + 17.0:.6f}"
    source_lines[frame_time + 1] = " ".join(values)
    refined_bvh = out_dir / "refined-input.bvh"
    refined_bvh.write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    refined_path = out_dir / "v325-refined.fbx"
    refined = convert(
        bvh_path=str(refined_bvh),
        character_fbx=args.character,
        out_path=str(refined_path),
        embed_textures=False,
    )
    assert refined.as_dict()["ok"] is True

    mirror_path = out_dir / "v325-refined-mirror.fbx"
    mirrored = convert(
        bvh_path=str(refined_bvh),
        character_fbx=args.character,
        out_path=str(mirror_path),
        mirror=True,
        embed_textures=False,
    )
    assert mirrored.as_dict()["ok"] is True
    assert len({_sha256(default_path), _sha256(refined_path), _sha256(mirror_path)}) == 3

    fallback_path = out_dir / "v324-kill-switch.fbx"
    fallback = convert(
        bvh_path=args.bvh,
        character_fbx=args.character,
        out_path=str(fallback_path),
        embed_textures=False,
        force_exact_v324=True,
    )
    fallback_payload = fallback.as_dict()
    assert fallback_payload["ok"] is True
    fallback_selection = fallback_payload["airborne_plantar_selection"]
    assert fallback_selection["status"] == "FORCED_EXACT_PARENT"
    assert fallback_selection["fallback_reason"] == "FORCED_EXACT_V324"
    assert fallback_selection["fallback_to_exact_v324"] is True
    assert fallback_selection["exact_parent_artifact_restored"] is True
    assert fallback_selection["parent_artifact_sha256"] == _sha256(fallback_path)

    payload = {
        "ok": True,
        "default_artifact_sha256": _sha256(default_path),
        "default_selection": selection,
        "refined_artifact_sha256": _sha256(refined_path),
        "mirror_artifact_sha256": _sha256(mirror_path),
        "fallback_artifact_sha256": _sha256(fallback_path),
        "fallback_selection": fallback_selection,
    }
    Path(args.json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[OK] V3.2.5 runtime + V3.2.4 kill switch")
    return 0


if __name__ == "__main__":
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(raw))
