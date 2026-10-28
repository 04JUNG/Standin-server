"""Assemble an app resource from a vetted Blender distribution, never system Python.

Builder input is an unpacked official Blender 5.2.0 distribution. No runtime is
downloaded or installed on an end user's machine. This creates a review/test
candidate; app signing, source/license delivery and OS qualification are release
gates, not implied by a successful directory copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def build(distribution, output, target, cs3c_modeler_exe=None):
    distribution, output = Path(distribution).resolve(), Path(output).absolute()
    if output.exists():
        raise ValueError("Output must be a new directory")
    if target in {"darwin-arm64", "darwin-x64"}:
        runtime_name = "Blender.app"
        blender = "runtime/Blender.app/Contents/MacOS/Blender"
        python = "runtime/Blender.app/Contents/Resources/5.2/python/bin/python3.13"
        launcher = "standin-local-converter"
    elif target == "windows-x64":
        runtime_name = "blender"
        blender = "runtime/blender/blender.exe"
        python = "runtime/blender/5.2/python/bin/python.exe"
        launcher = "standin-local-converter.cmd"
    else:
        raise ValueError("Unsupported package target")
    for relative in (blender, python):
        if not (distribution / Path(relative).relative_to(Path("runtime") / runtime_name)).is_file():
            raise ValueError("Blender 5.2.0 distribution is missing the required executable")
    output.mkdir(parents=True)
    shutil.copytree(distribution, output / "runtime" / runtime_name, symlinks=True)
    shutil.copytree(ROOT / "converter", output / "converter",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "hand_thumbs", "cs3c_decoder_data.json"))
    cs3c = {"enabled": False}
    if cs3c_modeler_exe:
        from converter.local.cs3c import build_decoder_resource
        cs3c = {"enabled": True, **build_decoder_resource(cs3c_modeler_exe,
                    output / "converter/local/cs3c_decoder_data.json")}
    shutil.copy2(ROOT / "docs/personal-body/converter-client-contract.v1.json", output)
    shutil.copy2(ROOT / "LICENSE", output / "CONVERTER_LICENSE.txt")
    shutil.copy2(Path(__file__).with_name("runtime_entry.py"), output / "entry.py")
    if launcher.endswith(".cmd"):
        text = '@echo off\r\nset "PYTHONHOME="\r\nset "PYTHONPATH="\r\nset "PYTHONNOUSERSITE=1"\r\nset "PYTHONDONTWRITEBYTECODE=1"\r\n"%~dp0' + python.replace("/", "\\") + '" -B -I "%~dp0entry.py" %*\r\n'
    else:
        text = '#!/bin/sh\nset -eu\nHERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nunset PYTHONHOME PYTHONPATH\nexport PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1\nexec "$HERE/' + python + '" -B -I "$HERE/entry.py" "$@"\n'
    (output / launcher).write_text(text)
    (output / launcher).chmod(0o755)
    files = {}
    byte_count = 0
    for path in sorted(output.rglob("*")):
        if path.is_file() and not path.is_symlink():
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024**2), b""):
                    byte_count += len(chunk)
                    h.update(chunk)
            files[path.relative_to(output).as_posix()] = h.hexdigest()
    manifest = {"schemaVersion": 1, "protocolVersion": "standin.local-converter/1", "target": target,
                "launcher": launcher, "blender": blender, "python": python, "blenderVersion": "5.2.0",
                "files": files, "uncompressedBytes": byte_count,
                "requiresUserInstalledRuntime": False, "releaseReady": False,
                "cs3c": cs3c,
                "releaseGates": ["macOS signing/notarization and Windows signing/qualification",
                                 "Blender corresponding source and complete third-party notices delivery",
                                 "CS3C redistributable decoder and appearance validation",
                                 "actual character compatibility and CSP acceptance"]}
    (output / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=["darwin-arm64", "darwin-x64", "windows-x64"], required=True)
    parser.add_argument("--cs3c-modeler-exe", type=Path,
                        help="Reviewed build-only data source; never executed or copied into runtime")
    args = parser.parse_args()
    result = build(args.distribution, args.output, args.target, args.cs3c_modeler_exe)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
