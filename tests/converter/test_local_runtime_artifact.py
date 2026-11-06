"""Synthetic package headers and fail-closed integrity; no OS execution claim."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.local_converter.runtime_entry import verify_artifact, verify_executable_target
from scripts.local_converter.package_runtime import build


def executable(target):
    header = bytearray(256)
    if target.startswith("darwin-"):
        header[:4] = (0xfeedfacf).to_bytes(4, "little")
        header[4:8] = (0x0100000c if target == "darwin-arm64" else 0x01000007).to_bytes(4, "little")
        header[12:16] = (2).to_bytes(4, "little")
    else:
        header[:2] = b"MZ"
        header[60:64] = (64).to_bytes(4, "little")
        header[64:70] = b"PE\0\0\x64\x86"
        header[88:90] = b"\x0b\x02"
    return bytes(header)


def fixture(root, target="darwin-arm64"):
    launcher = "standin-local-converter.cmd" if target == "windows-x64" else "standin-local-converter"
    content = {launcher: b"launcher", "entry.py": b"entry", "converter-client-contract.v1.json": b"{}",
               "blender": executable(target), "python": executable(target)}
    for name, data in content.items():
        (root/name).write_bytes(data)
    manifest = {"schemaVersion": 1, "protocolVersion": "standin.local-converter/1", "target": target,
                "requiresUserInstalledRuntime": False, "launcher": launcher, "blender": "blender", "python": "python",
                "files": {k: hashlib.sha256(v).hexdigest() for k, v in content.items()}}
    (root/"runtime-manifest.json").write_text(json.dumps(manifest))
    return manifest


class RuntimeArtifactTests(unittest.TestCase):
    def test_native_headers_and_host_architecture_are_independent_checks(self):
        for target, system, machine in [("darwin-arm64", "Darwin", "arm64"),
                                        ("darwin-x64", "Darwin", "x86_64"),
                                        ("windows-x64", "Windows", "AMD64")]:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = fixture(root, target)
                with patch("platform.system", return_value=system), patch("platform.machine", return_value=machine):
                    verify_artifact(root, manifest)
                with patch("platform.system", return_value=system), patch("platform.machine", return_value="unsupported"):
                    with self.assertRaisesRegex(RuntimeError, "OS/architecture"):
                        verify_artifact(root, manifest)
                other = executable("darwin-x64" if target == "darwin-arm64" else "darwin-arm64")
                (root/"python").write_bytes(other)
                manifest["files"]["python"] = hashlib.sha256(other).hexdigest()
                with self.assertRaisesRegex(ValueError, "architecture|Windows"):
                    verify_executable_target(root/"python", target)

    def test_exact_tree_rejects_stale_decoder_missing_tamper_and_links(self):
        with tempfile.TemporaryDirectory() as tmp, patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            root = Path(tmp)
            manifest = fixture(root)
            (root/"stale-decoder.py").write_text("extra")
            with self.assertRaisesRegex(RuntimeError, "Unlisted"):
                verify_artifact(root, manifest)
            (root/"stale-decoder.py").unlink()
            (root/"entry.py").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing"):
                verify_artifact(root, manifest)
            (root/"entry.py").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                verify_artifact(root, manifest)
            (root/"entry.py").write_bytes(b"entry")
            if os.name != "nt":
                (root/"entry.py").unlink()
                (root/"entry.py").symlink_to(root/"converter-client-contract.v1.json")
                with self.assertRaisesRegex(RuntimeError, "symlink"):
                    verify_artifact(root, manifest)

    def test_manifest_paths_and_required_hash_entries_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp, patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            root = Path(tmp)
            for relative in ["../escape", "/absolute", "a/../entry", "a\\b", "C:/runtime"]:
                manifest = fixture(root)
                manifest["files"][relative] = "a" * 64
                with self.subTest(relative=relative), self.assertRaisesRegex(RuntimeError, "manifest entry"):
                    verify_artifact(root, manifest)
            manifest = fixture(root)
            del manifest["files"]["python"]
            with self.assertRaisesRegex(RuntimeError, "not hash pinned"):
                verify_artifact(root, manifest)

    def test_builder_rejects_mislabeled_distribution_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            distribution = root/"Blender.app"
            for relative in ["Contents/MacOS/Blender", "Contents/Resources/5.2/python/bin/python3.13"]:
                file = distribution/relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(executable("darwin-arm64"))
            with self.assertRaisesRegex(ValueError, "architecture"):
                build(distribution, root/"rejected", "darwin-x64")
            self.assertFalse((root/"rejected").exists())
