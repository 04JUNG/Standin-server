"""Server-authored final-byte inspection job; one isolated Blender scene."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converter.selected_snapshot import encode, inspect_final, revision, sha
from converter.worker import _frozen_lineage
from converter.protocol import EXPECTED_BLENDER_BUILD_HASH, EXPECTED_BLENDER_VERSION


def main():
    import bpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    path = Path(args.job).resolve(strict=True)
    root = path.parent
    data = path.read_bytes()
    job = json.loads(data)
    if (bpy.app.build_hash.decode() != EXPECTED_BLENDER_BUILD_HASH
            or not bpy.app.version_string.startswith(EXPECTED_BLENDER_VERSION)):
        raise ValueError("frozen Blender runtime mismatch")
    _frozen_lineage()
    if job["revision"] != revision() or sha((root / "final.fbx").read_bytes()) != job["artifact_sha256"]:
        raise ValueError("candidate inspection lineage mismatch")
    report = inspect_final(root / "final.fbx", root / "preview.glb", job["binding"],
                           full_body=job["binding"]["output_scope"] == "full")
    report.update(job_sha256=sha(data), revision=revision(), ok=True)
    (root / "report.json").write_bytes(encode(report))


if __name__ == "__main__":
    main()
