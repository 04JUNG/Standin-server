"""Packaged entry: verify shipped bytes before starting the local IPC host."""
import hashlib
import json
from pathlib import Path
import platform
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "runtime-manifest.json").read_text())
    system = platform.system()
    target = manifest["target"]
    if (target.startswith("darwin-") and system != "Darwin") or (target.startswith("windows-") and system != "Windows"):
        raise RuntimeError("Local converter package does not match this OS")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise RuntimeError("Invalid runtime manifest path")
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024**2), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise RuntimeError("Local converter package integrity check failed")
    sys.path.insert(0, str(root))
    from converter.local.host import main as run_host
    # The client controls only --storage. It cannot select a different executable.
    if len(sys.argv) != 3 or sys.argv[1] != "--storage":
        raise RuntimeError("Expected --storage <absolute app-owned directory>")
    sys.argv.extend(["--blender", str(root / manifest["blender"])])
    run_host()


if __name__ == "__main__":
    main()
