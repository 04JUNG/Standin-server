"""Exercise a packaged runtime with real FBX/BVH bytes over its public IPC.

No assets are bundled with this test. Supply locally permitted regression files.
Raw device paths stay in the optional local transcript, not the portable report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

PROTOCOL = "standin.local-converter/1"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Session:
    def __init__(self, launcher, storage):
        self.events = queue.Queue()
        self.transcript = []
        self.process = subprocess.Popen([str(launcher), "--storage", str(storage)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        def read():
            for line in self.process.stdout:
                self.events.put(json.loads(line))
            self.events.put(None)
        threading.Thread(target=read, daemon=True).start()

    def send(self, action, **fields):
        rid = uuid.uuid4().hex
        raw = {"protocolVersion": PROTOCOL, "requestId": rid, "action": action, **fields}
        self.transcript.append({"request": raw})
        self.process.stdin.write(json.dumps(raw) + "\n")
        self.process.stdin.flush()
        return rid

    def receive(self, rid, error=None):
        while True:
            event = self.events.get(timeout=150)
            assert event is not None, "Runtime exited before a terminal event"
            self.transcript.append({"event": event})
            assert event["protocolVersion"] == PROTOCOL and event["requestId"] == rid, event
            if event["event"] == "progress":
                continue
            if error:
                assert event["event"] == "error" and event["error"]["code"] == error, event
                return event["error"]
            assert event["event"] == "result", event
            return event["result"]

    def request(self, action, error=None, **fields):
        return self.receive(self.send(action, **fields), error)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=20)
        assert self.process.returncode == 0


def verify(launcher, source, poses, cs3c, report_path, relocated=False):
    root = Path(tempfile.mkdtemp(prefix="standin-packaged-qa-", dir=report_path.parent))
    storage = root / "store"
    transcript = []
    source_hash = sha(source)
    start = time.monotonic()
    session = Session(launcher, storage)
    hello = session.request("hello")
    startup = time.monotonic() - start
    model = session.request("import", localModelId="regression", sourcePath=str(source), sourceHash=source_hash)
    assert model["validation"]["passed"] and model["sourceHash"] == source_hash
    assert model["capabilities"]["bodyPose"] is True
    if model["previewPath"]:
        assert sha(model["previewPath"]) == model["previewHash"]
    session.close()
    transcript += session.transcript
    # A new host must reuse a committed model without the original source location.
    session = Session(launcher, storage)
    reapplied_model = session.request("inspect", localModelId="regression")
    assert reapplied_model["normalizedHash"] == model["normalizedHash"]
    outputs = []
    for i, pose in enumerate(poses):
        fields = {"localModelId": "regression", "sourceHash": source_hash, "posePath": str(pose),
                  "poseHash": sha(pose), "selectionId": "local-regression-selection",
                  "hands": {"left": None, "right": None}}
        applied = session.request("apply", **fields)
        assert applied["validation"]["passed"] and not applied["cacheHit"]
        assert sha(applied["artifactPath"]) == applied["artifactHash"]
        cached = session.request("apply", **fields)
        assert cached["snapshotHash"] == applied["snapshotHash"] and cached["cacheHit"]
        changed_pose = root / f"changed-{i}.bvh"
        changed_pose.write_bytes(pose.read_bytes() + b"\n")
        session.request("apply", error="SOURCE_CHANGED", **{**fields, "posePath": str(changed_pose)})
        destination = root / f"export-{i}.fbx"
        exported = session.request("export", localModelId="regression", snapshotHash=applied["snapshotHash"],
                                   destinationPath=str(destination))
        assert sha(destination) == exported["artifactHash"] == applied["artifactHash"]
        session.request("export", error="DESTINATION_EXISTS", localModelId="regression",
                        snapshotHash=applied["snapshotHash"], destinationPath=str(destination))
        assert sha(destination) == exported["artifactHash"]
        session.request("apply", error="HAND_UNSUPPORTED", **{**fields, "hands": {"left": "fist", "right": None}})
        outputs.append({k: applied[k] for k in ("snapshotHash", "artifactHash", "validation", "metrics")}
                                | {"poseHash": sha(pose), "cacheReusePassed": True, "exportPassed": True})
    assert len({r["snapshotHash"] for r in outputs}) == len(outputs)
    session.request("export", error="SNAPSHOT_MISMATCH", localModelId="regression", snapshotHash="0"*64,
                    destinationPath=str(root / "stale.fbx"))
    session.request("import", error="MODEL_EXISTS", localModelId="regression", sourcePath=str(source), sourceHash=source_hash)
    if cs3c:
        session.request("import", error="CS3C_DECODER_UNAVAILABLE", localModelId="cs3c",
                        sourcePath=str(cs3c), sourceHash=sha(cs3c))
        session.request("inspect", error="MODEL_NOT_FOUND", localModelId="cs3c")
    import_id = session.send("import", localModelId="cancelled", sourcePath=str(source), sourceHash=source_hash)
    while True:
        event = session.events.get(timeout=10)
        session.transcript.append({"event": event})
        assert event and event["requestId"] == import_id and event["event"] == "progress", event
        if event["stage"] == "inspecting":
            break
    time.sleep(.5)  # Give the real Blender child time to enter model parsing.
    cancel_id = session.send("cancel", targetRequestId=import_id)
    terminals = {}
    while len(terminals) != 2:
        event = session.events.get(timeout=20)
        session.transcript.append({"event": event})
        assert event and event["requestId"] in {cancel_id, import_id}, event
        if event["event"] != "progress":
            terminals[event["requestId"]] = event
    assert terminals[cancel_id]["result"]["accepted"]
    assert terminals[import_id]["error"]["code"] == "CANCELLED"
    assert not (storage / "models/cancelled").exists()
    # EOF during an import must cancel the child and leave no partially registered model.
    session.close()
    transcript += session.transcript
    session = Session(launcher, storage)
    session.send("import", localModelId="interrupted", sourcePath=str(source), sourceHash=source_hash)
    session.close()
    transcript += session.transcript
    assert not (storage / "models/interrupted").exists()
    assert not list((storage / "staging").iterdir())
    session = Session(launcher, storage)
    assert session.request("inspect", localModelId="regression")["sourceHash"] == source_hash
    session.request("delete", localModelId="regression")
    session.request("inspect", error="MODEL_NOT_FOUND", localModelId="regression")
    session.close()
    transcript += session.transcript
    assert all((root / f"export-{i}.fbx").exists() for i in range(len(poses)))
    package = launcher.parent
    manifest = json.loads((package / "runtime-manifest.json").read_text())
    assert all(sha(package / rel) == expected for rel, expected in manifest["files"].items()), "Runtime mutated"
    extras = [p for p in package.rglob("*") if p.is_file() and not p.is_symlink()
              and p.relative_to(package).as_posix() not in manifest["files"] and p.name != "runtime-manifest.json"]
    assert not extras, "Runtime created untracked files"
    report = {"protocolVersion": PROTOCOL, "scope": "Packaged macOS public IPC with real FBX and library BVH; no BFF/CSP/Windows qualification",
              "startupSeconds": startup, "versions": hello["versions"], "sourceHash": source_hash,
              "registration": {k: model[k] for k in ("validation", "capabilities", "metrics")}, "outputs": outputs,
              "restartReusePassed": True, "staleSnapshotRejected": True, "changedPoseRejected": True,
              "noOverwritePassed": True, "activeCancellationPassed": True,
              "eofCancellationPassed": True, "deletePreservesExportsPassed": True,
              "cs3cFailClosedPassed": bool(cs3c), "allPassed": True,
              "packageIntegrityAfterRepeatedStarts": {"manifestFiles": len(manifest["files"]), "changed": 0, "extra": 0},
              "copiedWithoutPreservingTimestamps": relocated}
    report_path.write_text(json.dumps(report, indent=2))
    (root / "transcript.json").write_text(json.dumps(transcript, indent=2))
    print(json.dumps({"report": str(report_path), "localArtifacts": str(root), "allPassed": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pose", type=Path, action="append", required=True)
    parser.add_argument("--cs3c", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--relocate-to", type=Path,
                        help="Copy package without preserving timestamps, then test the copy (must be a new directory)")
    args = parser.parse_args()
    launcher = args.launcher.resolve()
    if args.relocate_to:
        relocated = args.relocate_to.resolve()
        shutil.copytree(launcher.parent, relocated, symlinks=True, copy_function=shutil.copy)
        launcher = relocated / launcher.name
    verify(launcher, args.source.resolve(), [p.resolve() for p in args.pose],
           args.cs3c.resolve() if args.cs3c else None, args.report.resolve(), bool(args.relocate_to))
