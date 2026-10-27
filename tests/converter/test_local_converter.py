"""IPC safety and real child-process failure/cancellation tests; no success mock."""
import json
import os
import subprocess
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from converter.local.contract import PROTOCOL, LIMITS, LocalError, digest, sha256, validate_request, snapshot, versions
from converter.local.host import Host, resident_bytes


class LocalConverterTests(unittest.TestCase):
    def test_untrusted_requests_fail_closed(self):
        base = {"protocolVersion": PROTOCOL, "requestId": "valid", "action": "hello"}
        for change in ({"protocolVersion": 2}, {"action": []}, {"action": "upload"},
                       {"requestId": "../escape"}, {"sourcePath": "/not-allowed"}):
            with self.subTest(change=change), self.assertRaises(LocalError):
                validate_request({**base, **change})

    def test_snapshot_changes_with_model_pose_hands_and_core(self):
        model = {"localModelId": "one", "sourceHash": "a" * 64, "versions": versions()}
        base = digest(snapshot(model, "b" * 64, {"left": None, "right": None}))
        variants = [snapshot({**model, "sourceHash": "c" * 64}, "b" * 64, {"left": None, "right": None}),
                    snapshot(model, "c" * 64, {"left": None, "right": None}),
                    snapshot(model, "b" * 64, {"left": "fist", "right": None}),
                    snapshot({**model, "versions": {**model["versions"], "converterFingerprint": "d"*64}},
                             "b" * 64, {"left": None, "right": None})]
        self.assertTrue(all(digest(v) != base for v in variants))

    def test_no_personal_finger_support_is_invented(self):
        request = {"protocolVersion": PROTOCOL, "requestId": "a", "action": "apply", "localModelId": "m",
                   "sourceHash": "a"*64, "poseHash": "b"*64, "posePath": "/pose.bvh", "selectionId": "selection",
                   "hands": {"left": "fist", "right": None}}
        with self.assertRaises(LocalError) as error:
            validate_request(request)
        self.assertEqual(error.exception.code, "HAND_UNSUPPORTED")

    def test_source_snapshot_detects_file_change_and_cleans_failed_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.fbx"
            source.write_bytes(b"source")
            events = []
            host = Host(root / "store", root / "missing-runtime", events.append)
            try:
                host.submit({"protocolVersion": PROTOCOL, "requestId": "import", "action": "import",
                             "localModelId": "m", "sourcePath": str(source), "sourceHash": "0"*64})
                host.pool.submit(lambda: None).result(timeout=5)
                self.assertEqual(events[-1]["error"]["code"], "SOURCE_CHANGED")
                self.assertEqual(list((root / "store/models").iterdir()), [])
                self.assertEqual(list((root / "store/staging").iterdir()), [])
                self.assertEqual(source.read_bytes(), b"source")
            finally:
                host.close()

    def test_missing_runtime_never_registers_a_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.fbx"
            source.write_bytes(b"source")
            events = []
            host = Host(root / "store", root / "missing-runtime", events.append)
            try:
                host.submit({"protocolVersion": PROTOCOL, "requestId": "import", "action": "import",
                             "localModelId": "m", "sourcePath": str(source), "sourceHash": sha256(source)})
                host.pool.submit(lambda: None).result(timeout=5)
                self.assertEqual(events[-1]["error"]["code"], "RUNTIME_UNAVAILABLE")
                self.assertFalse((root / "store/models/m").exists())
                self.assertEqual(list((root / "store/staging").iterdir()), [])
            finally:
                host.close()

    @unittest.skipIf(os.name == "nt", "POSIX failure harness; Windows package requires native QA")
    def test_cancel_timeout_and_memory_limit_kill_real_children(self):
        for reason in ("CANCELLED", "TIMEOUT", "MEMORY_LIMIT"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                executable = root / "slow-runtime"
                executable.write_text("#!/bin/sh\nexec sleep 30\n")
                executable.chmod(0o755)
                source = root / "source.fbx"
                source.write_bytes(b"source")
                events, pids = [], []
                host = Host(root / "store", executable, events.append)
                observed = threading.Event()
                def observe(pid):
                    pids.append(pid)
                    observed.set()
                    return resident_bytes(pid)
                popen = subprocess.Popen
                def start(command, *args, **kwargs):
                    if command[0] == str(executable):
                        child_env = kwargs["env"]
                        self.assertNotIn("STANDIN_WRIST_TRANSFER", child_env)
                        self.assertNotIn("STANDIN_WRIST_TWIST_FOREARM_SHARE", child_env)
                        self.assertNotIn("PYTHONPATH", child_env)
                        self.assertIn("--python-use-system-env", command)
                        self.assertEqual(child_env["PYTHONDONTWRITEBYTECODE"], "1")
                        self.assertEqual(child_env["PYTHONNOUSERSITE"], "1")
                        self.assertEqual(child_env["PYTHONSAFEPATH"], "1")
                    return popen(command, *args, **kwargs)
                limits = {**LIMITS, "workerSeconds": .2 if reason == "TIMEOUT" else 10,
                          "workerMemoryBytes": 1 if reason == "MEMORY_LIMIT" else LIMITS["workerMemoryBytes"]}
                try:
                    with (patch.dict(LIMITS, limits), patch("converter.local.host.resident_bytes", observe),
                          patch.dict(os.environ, {"STANDIN_WRIST_TRANSFER": "0", "PYTHONPATH": "/untrusted",
                                                  "STANDIN_WRIST_TWIST_FOREARM_SHARE": "not-a-number"}),
                          patch("converter.local.host.subprocess.Popen", start)):
                        host.submit({"protocolVersion": PROTOCOL, "requestId": "import", "action": "import",
                                     "localModelId": "m", "sourcePath": str(source), "sourceHash": sha256(source)})
                        self.assertTrue(observed.wait(3))
                        if reason == "CANCELLED":
                            host.submit({"protocolVersion": PROTOCOL, "requestId": "cancel", "action": "cancel",
                                         "targetRequestId": "import"})
                        host.pool.submit(lambda: None).result(timeout=5)
                    self.assertEqual(events[-1]["error"]["code"], reason)
                    self.assertFalse((root / "store/models/m").exists())
                    self.assertEqual(list((root / "store/staging").iterdir()), [])
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pids[0], 0)
                finally:
                    host.close()

    def test_second_host_cannot_remove_first_hosts_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = Host(root, root / "runtime", lambda _: None)
            marker = root / "staging/keep"
            marker.write_text("active")
            try:
                with self.assertRaises(LocalError) as error:
                    Host(root, root / "runtime", lambda _: None)
                self.assertEqual(error.exception.code, "BUSY")
                self.assertTrue(marker.exists())
            finally:
                host.close()

    def test_malformed_request_id_is_not_echoed_as_an_invalid_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            host = Host(tmp, Path(tmp) / "runtime", events.append)
            try:
                host.submit({"protocolVersion": PROTOCOL, "requestId": "../private/path", "action": "hello"})
                self.assertIsNone(events[-1]["requestId"])
                self.assertEqual(events[-1]["error"]["code"], "INVALID_REQUEST")
            finally:
                host.close()


if __name__ == "__main__":
    unittest.main()
