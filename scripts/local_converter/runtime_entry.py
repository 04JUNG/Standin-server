"""Packaged entry: verify shipped bytes before starting the local IPC host."""
import hashlib
import json
from pathlib import Path
import platform
import sys


TARGETS = {
    "darwin-arm64": ("Darwin", {"arm64", "aarch64"}),
    "darwin-x64": ("Darwin", {"x86_64", "amd64"}),
    "windows-x64": ("Windows", {"amd64", "x86_64"}),
}


def verify_executable_target(path, target):
    """Read native headers; a directory label is not architecture evidence."""
    with Path(path).open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64:
            raise ValueError("Truncated runtime executable")
        if target in {"darwin-arm64", "darwin-x64"}:
            cpu = 0x0100000c if target == "darwin-arm64" else 0x01000007
            if (int.from_bytes(header[:4], "little") != 0xfeedfacf or
                    int.from_bytes(header[4:8], "little") != cpu or
                    int.from_bytes(header[12:16], "little") != 2):
                raise ValueError("Runtime executable architecture does not match target")
        elif target == "windows-x64":
            offset = int.from_bytes(header[60:64], "little")
            if header[:2] != b"MZ" or not 64 <= offset <= 1024**2:
                raise ValueError("Invalid Windows runtime executable")
            stream.seek(offset)
            pe = stream.read(26)
            if len(pe) != 26 or pe[:6] != b"PE\0\0\x64\x86" or pe[24:26] != b"\x0b\x02":
                raise ValueError("Runtime executable architecture does not match target")
        else:
            raise ValueError("Unsupported runtime target")


def verify_artifact(root, manifest):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("Runtime must be a real directory")
    target = manifest.get("target")
    host = TARGETS.get(target)
    if not host or platform.system() != host[0] or platform.machine().lower() not in host[1]:
        raise RuntimeError("Local converter package does not match this OS/architecture")
    files = manifest.get("files")
    launcher = "standin-local-converter.cmd" if target == "windows-x64" else "standin-local-converter"
    if (manifest.get("schemaVersion") != 1 or manifest.get("protocolVersion") != "standin.local-converter/1" or
            manifest.get("requiresUserInstalledRuntime") is not False or manifest.get("launcher") != launcher or
            not isinstance(files, dict) or not files):
        raise RuntimeError("Invalid runtime manifest")
    for required in (launcher, "entry.py", "converter-client-contract.v1.json", manifest.get("blender"), manifest.get("python")):
        if not isinstance(required, str) or required not in files:
            raise RuntimeError("Required runtime entry is not hash pinned")
    for relative, expected in files.items():
        if (not relative or "\\" in relative or ":" in relative or
                any(part in {"", ".", ".."} for part in relative.split("/")) or
                relative == "runtime-manifest.json" or not isinstance(expected, str) or len(expected) != 64 or
                any(char not in "0123456789abcdef" for char in expected)):
            raise RuntimeError("Invalid runtime manifest entry")
    observed = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError("Runtime symlinks are not supported")
        if path.is_dir():
            if not any(name.startswith(relative + "/") for name in files):
                raise RuntimeError("Unlisted runtime directory")
            continue
        if not path.is_file():
            raise RuntimeError("Unsupported runtime file type")
        if relative == "runtime-manifest.json":
            continue
        if relative not in files:
            raise RuntimeError("Unlisted runtime file")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != files[relative]:
            raise RuntimeError("Local converter package integrity check failed")
        observed.add(relative)
    if observed != set(files):
        raise RuntimeError("Runtime is missing listed files")
    for field in ("blender", "python"):
        verify_executable_target(root / manifest[field], target)


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "runtime-manifest.json").read_text())
    verify_artifact(root, manifest)
    sys.path.insert(0, str(root))
    from converter.local.host import main as run_host
    # The client controls only --storage. It cannot select a different executable.
    if len(sys.argv) != 3 or sys.argv[1] != "--storage":
        raise RuntimeError("Expected --storage <absolute app-owned directory>")
    sys.argv.extend(["--blender", str(root / manifest["blender"])])
    run_host()


if __name__ == "__main__":
    main()
