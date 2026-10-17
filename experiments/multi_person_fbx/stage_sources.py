"""Stage audited experiment scripts into a private workspace, without input data.

The source snapshots remain byte-for-byte evidence of the local experiment.
Only executable/dependency paths are substituted in the staged copies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def stage(config: dict, *, validate_only: bool = False) -> list[dict]:
    required = (
        "workspace", "repo", "routing_repo", "python", "layout_python",
        "blender", "node", "three_dir", "humanart_manifest", "font", "label_font",
    )
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError("Missing runtime paths: " + ", ".join(missing))
    paths = {key: Path(config[key]).expanduser().resolve() for key in required}
    for key in required[1:]:
        if not paths[key].exists():
            raise ValueError(f"Missing {key}: {paths[key]}")
    for file in ("converter/scene.py", "src/bvh.py"):
        if not (paths["repo"] / file).is_file():
            raise ValueError(f"Converter checkout missing {file}")
    if not (paths["routing_repo"] / "src/semantic_adapter_285.py").is_file():
        raise ValueError("routing_repo must contain the semantic285 adapter")
    manifest = json.loads((HERE / "sources-manifest.json").read_text())
    replacements = {
        "WORK/'.worktrees/standin-multi-person-fbx'": f"Path({str(paths['repo'])!r})",
        "WORK/'.worktrees/standin-fullbody-routing'": f"Path({str(paths['routing_repo'])!r})",
        "WORK/'.worktrees/standin-fullbody-routing/src'": f"Path({str(paths['routing_repo'] / 'src')!r})",
        "WORK/'.worktrees/standin-fullbody-routing/config/refine_pose_quarantine.v1.json'":
            f"Path({str(paths['routing_repo'] / 'config/refine_pose_quarantine.v1.json')!r})",
        "WORK/'.worktrees/standin-dynamic-routing/tmp/humanart-local-bundle/manifest.json'":
            f"Path({str(paths['humanart_manifest'])!r})",
        "'/private/tmp/standin-04j289-venv/bin/python'": repr(str(paths["python"])),
        "str(WORK/'tmp/rtmpose_eval_venv/bin/python')": repr(str(paths["layout_python"])),
        "'/private/tmp/standin-complement-blender-KDq2Ja/Blender.app/Contents/MacOS/Blender'": repr(str(paths["blender"])),
        "'/Users/dw/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'": repr(str(paths["node"])),
        "'/System/Library/Fonts/AppleSDGothicNeo.ttc'": repr(str(paths["font"])),
        "'/System/Library/Fonts/Helvetica.ttc'": repr(str(paths["label_font"])),
        "/private/tmp/standin-04j289-three/node_modules/three/": paths["three_dir"].as_uri() + "/",
    }
    jobs = []
    for row in manifest["files"]:
        source = HERE / row["snapshot"]
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Snapshot checksum changed: {source}")
        destination = paths["workspace"] / row["source"]
        if destination.exists():
            raise ValueError(f"Refusing to overwrite existing script: {destination}")
        text = raw.decode("utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        if source.suffix == ".py":
            compile(text, str(destination), "exec")
        jobs.append((destination, text, row["sha256"]))
    # Validate every source and collision before creating any file.
    if not validate_only:
        for destination, text, _ in jobs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as output:
                output.write(text)
    return [dict(path=str(path), source_sha256=sha) for path, _, sha in jobs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.runtime.read_text())
    try:
        rows = stage(config, validate_only=args.validate_only)
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(dict(validated=True, written=not args.validate_only, scripts=rows), indent=2))


if __name__ == "__main__":
    main()
