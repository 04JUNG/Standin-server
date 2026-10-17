"""Offline equivalent of /compose-scene for final rigged-rest FBXs."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.scene_contract import MAX_FBX_BYTES, validate_manifest
from converter_api.runner import BlenderRunner
from converter_api.scene_routes import scene_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fbx", type=Path, action="append", required=True,
                        help="repeat in manifest person order")
    parser.add_argument("--out", type=Path, required=True, help="destination ZIP (must not exist)")
    args = parser.parse_args()
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    if args.out.exists():
        parser.error("output already exists")
    if len(args.fbx) != len(manifest["persons"]):
        parser.error("one --fbx per person is required")
    if any(path.stat().st_size > MAX_FBX_BYTES for path in args.fbx):
        parser.error("FBX exceeds size limit")
    result = BlenderRunner().compose_scene(manifest=manifest, fbx_bytes=[p.read_bytes() for p in args.fbx])
    data = scene_bundle(manifest, result)
    with args.out.open("xb") as handle:
        handle.write(data)
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
