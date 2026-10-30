"""Device-only NDJSON host; Blender is always a cancellable child process.

Run with the packaged Python, never the UI process. No network libraries, HTTP
listeners, server fallback, automatic asset discovery or external commands from
input files are used here.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.local.contract import (PROTOCOL, LIMITS, LocalError, absolute_file,
                                      digest, identifier, sha256, snapshot, validate_request, versions)


def resident_bytes(pid):
    if sys.platform == "darwin":
        # Darwin's proc_taskinfo layout is defined in <sys/proc_info.h>.
        # Avoid spawning ps repeatedly while a conversion is running.
        class TaskInfo(ctypes.Structure):
            _fields_ = [("virtual", ctypes.c_uint64), ("resident", ctypes.c_uint64),
                        ("times", ctypes.c_uint64 * 4), ("counters", ctypes.c_int32 * 12)]
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        query = lib.proc_pidinfo
        query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        info = TaskInfo()
        size = query(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        if size != ctypes.sizeof(info):
            if ctypes.get_errno() == 3:  # ESRCH: worker exited between polls.
                return 0
            raise LocalError("RUNTIME_UNAVAILABLE", "Cannot enforce the worker memory limit")
        return info.resident
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x410, False, pid)
        query = kernel.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        query.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        try:
            if not handle or not query(handle, ctypes.byref(counters), counters.cb):
                raise LocalError("RUNTIME_UNAVAILABLE", "Cannot enforce the worker memory limit")
            return counters.WorkingSetSize
        finally:
            if handle:
                kernel.CloseHandle(handle)
    read = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(pid)], capture_output=True, timeout=2)
    if read.returncode:
        return 0  # A process that just exited has no resident memory.
    try:
        return int(read.stdout.strip()) * 1024
    except ValueError:
        return 0


def stop_process(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=10)


class Host:
    def __init__(self, storage, blender, emit):
        self.root = Path(storage).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blender = str(Path(blender).resolve())
        self.emit = emit
        self.lock = threading.RLock()
        self.active = None
        self.committed = False
        self.cancelled = threading.Event()
        self.seen = set()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._storage_lock = (self.root / ".host-lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self._storage_lock.write(b"0")
                self._storage_lock.flush()
                self._storage_lock.seek(0)
                msvcrt.locking(self._storage_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._storage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._storage_lock.close()
            raise LocalError("BUSY", "Another local converter owns this storage") from None
        (self.root / "models").mkdir(exist_ok=True, mode=0o700)
        self.staging = self.root / "staging"
        self.staging.mkdir(exist_ok=True, mode=0o700)
        # Only our staging is disposable. Completed cards and user exports survive.
        for child in self.staging.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    def event(self, request_id, kind, **body):
        self.emit({"protocolVersion": PROTOCOL, "requestId": request_id, "event": kind, **body})

    def progress(self, request_id, stage, progress):
        self.event(request_id, "progress", stage=stage, progress=progress)

    def check_cancel(self):
        if self.cancelled.is_set():
            raise LocalError("CANCELLED", "Local operation cancelled")

    def submit(self, raw):
        request_id = raw.get("requestId") if isinstance(raw, dict) else None
        try:
            identifier(request_id)
        except LocalError:
            request_id = None
        try:
            req = validate_request(raw)
            with self.lock:
                if request_id in self.seen:
                    raise LocalError("INVALID_REQUEST", "Request IDs must be unique for this session")
                if len(self.seen) >= 10000:
                    raise LocalError("LIMIT_EXCEEDED", "Reconnect the local session after 10000 requests")
                self.seen.add(request_id)
                action = req["action"]
                if action == "hello":
                    from converter.local.cs3c import decoder_available
                    self.event(request_id, "result", result={"protocolVersion": PROTOCOL,
                               "versions": versions(), "limits": LIMITS,
                               "formats": {"fbx": True, "cs3c": decoder_available()}, "hands": False})
                    return
                if action == "cancel":
                    accepted = self.active == req["targetRequestId"] and not self.committed
                    if accepted:
                        self.cancelled.set()
                    self.event(request_id, "result", result={"targetRequestId": req["targetRequestId"],
                                                            "accepted": accepted})
                    return
                if self.active:
                    raise LocalError("BUSY", "A local operation is already in progress")
                self.active = request_id
                self.committed = False
                self.cancelled.clear()
                self.pool.submit(self.execute, req)
        except LocalError as exc:
            self.event(request_id, "error", error=exc.payload())

    def model_dir(self, model_id):
        folder = self.root / "models" / model_id
        if folder.is_symlink():
            raise LocalError("IO_ERROR", "Invalid local model storage")
        return folder

    def load_model(self, model_id):
        folder = self.model_dir(model_id)
        try:
            manifest = json.loads((folder / "manifest.json").read_text())
        except (OSError, ValueError):
            raise LocalError("MODEL_NOT_FOUND", "Registered model was not found") from None
        if manifest["versions"] != versions():
            raise LocalError("RUNTIME_MISMATCH", "Reimport this model with the current converter")
        if sha256(folder / "character.blend") != manifest["normalizedHash"]:
            raise LocalError("SOURCE_CHANGED", "Stored character integrity check failed")
        return folder, manifest

    def copy_input(self, source, target, expected_hash, limit):
        h = __import__("hashlib").sha256()
        total = 0
        with source.open("rb") as src, target.open("xb") as dst:
            while chunk := src.read(1024**2):
                self.check_cancel()
                total += len(chunk)
                if total > limit:
                    raise LocalError("SOURCE_TOO_LARGE", "Input changed or exceeded the file limit")
                dst.write(chunk)
                h.update(chunk)
        if h.hexdigest() != expected_hash:
            raise LocalError("SOURCE_CHANGED", "Selected file changed before the local snapshot was made")

    def worker(self, job, request_id):
        work = Path(job["workDir"])
        job["resultPath"] = str(work / "worker-result.json")
        job_path = work / "job.json"
        job_path.write_text(json.dumps(job))
        # Isolate config, scripts, extensions and auto-exec from a user's Blender installation.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("PYTHON", "BLENDER", "STANDIN_", "CONVERTER_"))}
        env.update({"BLENDER_USER_RESOURCES": str(work / "blender-user"),
                    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONSAFEPATH": "1"})
        script = Path(__file__).with_name("blender_job.py")
        try:
            proc = subprocess.Popen([self.blender, "--background", "--factory-startup", "--disable-autoexec",
                                     # Blender otherwise ignores Python environment controls during startup.
                                     # Only our three sanitized PYTHON* settings above are inherited.
                                     "--python-use-system-env",
                                     "--threads", "2", "--python", str(script), "--", str(job_path)],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    env=env, start_new_session=os.name != "nt",
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        except OSError:
            raise LocalError("RUNTIME_UNAVAILABLE", "Bundled local runtime could not start") from None
        started = time.monotonic()
        peak = 0
        try:
            while proc.poll() is None:
                self.check_cancel()
                if time.monotonic() - started > LIMITS["workerSeconds"]:
                    raise LocalError("TIMEOUT", "Local conversion exceeded the time limit")
                peak = max(peak, resident_bytes(proc.pid))
                if peak > LIMITS["workerMemoryBytes"]:
                    raise LocalError("MEMORY_LIMIT", "Local conversion exceeded the memory limit")
                self.cancelled.wait(.1)
        finally:
            stop_process(proc)
        self.check_cancel()
        try:
            result = json.loads(Path(job["resultPath"]).read_text())
        except (OSError, ValueError):
            raise LocalError("CONVERSION_FAILED", "Local runtime exited without a valid result") from None
        if proc.returncode != 0 or not result.get("ok"):
            err = result.get("error", {})
            raise LocalError(err.get("code", "CONVERSION_FAILED"), err.get("message", "Local conversion failed"), err.get("details"))
        result["result"]["metrics"] = {"elapsedSeconds": time.monotonic() - started, "peakResidentBytes": peak}
        return result["result"]

    def perform(self, req, work):
        action, model_id, rid = req["action"], req["localModelId"], req["requestId"]
        folder = self.model_dir(model_id)
        if action == "import":
            if folder.exists():
                raise LocalError("MODEL_EXISTS", "File replacement requires a new local model ID")
            source = absolute_file(req["sourcePath"], {".fbx", ".cs3c"}, LIMITS["sourceBytes"])
            self.progress(rid, "copying", .05)
            copied = work / ("original" + source.suffix.lower())
            self.copy_input(source, copied, req["sourceHash"], LIMITS["sourceBytes"])
            self.progress(rid, "inspecting", .2)
            manifest = self.worker({"action": "import", "workDir": str(work), "sourcePath": str(copied),
                                    "localModelId": model_id, "sourceHash": req["sourceHash"]}, rid)
            manifest["normalizedPath"] = str(folder / "character.blend")
            manifest["sourcePath"] = str(folder / copied.name)
            manifest["previewPath"] = str(folder / "preview.glb") if manifest["previewPath"] else None
            self.progress(rid, "committing", .95)
            (work / "manifest.json").write_text(json.dumps(manifest))
            # No raw job JSON, temporary FBX, runtime resources or source paths in committed caches.
            for child in list(work.iterdir()):
                if child.name not in {copied.name, "character.blend", "manifest.json", "preview.glb"}:
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
            with self.lock:
                self.check_cancel()
                work.rename(folder)
                self.committed = True
            return manifest
        if action == "delete":
            if not folder.is_dir():
                raise LocalError("MODEL_NOT_FOUND", "Registered model was not found")
            with self.lock:
                self.check_cancel()
                folder.rename(work / "deleted")
                self.committed = True
            return {"deleted": True}
        folder, manifest = self.load_model(model_id)
        if action == "inspect":
            return manifest
        if action == "export":
            cache = folder / "results" / req["snapshotHash"]
            try:
                result = json.loads((cache / "result.json").read_text())
            except (OSError, ValueError):
                raise LocalError("SNAPSHOT_MISMATCH", "This model has no verified result for the selected snapshot") from None
            artifact = cache / "result.fbx"
            if result["snapshotHash"] != req["snapshotHash"] or sha256(artifact) != result["artifactHash"]:
                raise LocalError("SNAPSHOT_MISMATCH", "Verified output integrity check failed")
            value = req["destinationPath"]
            if not isinstance(value, str) or not Path(value).is_absolute() or Path(value).suffix.lower() != ".fbx":
                raise LocalError("INVALID_REQUEST", "Choose an absolute FBX destination")
            dest = Path(value)
            if dest.exists() or dest.is_symlink():
                raise LocalError("DESTINATION_EXISTS", "The selected destination already exists")
            fd, name = tempfile.mkstemp(prefix=".standin-export-", dir=dest.parent)
            os.close(fd)
            tmp = Path(name)
            try:
                shutil.copyfile(artifact, tmp)
                with self.lock:
                    self.check_cancel()
                    # Atomic publish without overwriting a pre-existing user file.
                    os.link(tmp, dest)
                    self.committed = True
            except FileExistsError:
                raise LocalError("DESTINATION_EXISTS", "The selected destination already exists") from None
            finally:
                tmp.unlink(missing_ok=True)
            return {"localModelId": model_id, "snapshotHash": req["snapshotHash"],
                    "artifactHash": result["artifactHash"], "destinationPath": str(dest)}
        if req["sourceHash"] != manifest["sourceHash"]:
            raise LocalError("SOURCE_CHANGED", "Selected model version has changed")
        pose = absolute_file(req["posePath"], {".bvh"}, LIMITS["poseBytes"])
        snap = snapshot(manifest, req["poseHash"], req["hands"])
        key = digest(snap)
        cache = folder / "results" / key
        if cache.exists():
            if sha256(pose) != req["poseHash"]:
                raise LocalError("SOURCE_CHANGED", "Selected pose changed before cache lookup")
            result = json.loads((cache / "result.json").read_text())
            if sha256(cache / "result.fbx") != result["artifactHash"]:
                raise LocalError("SNAPSHOT_MISMATCH", "Cached output integrity check failed")
            return {**result, "selectionId": req["selectionId"], "cacheHit": True}
        copied = work / "pose.bvh"
        self.copy_input(pose, copied, req["poseHash"], LIMITS["poseBytes"])
        self.progress(rid, "retargeting", .15)
        result = self.worker({"action": "apply", "workDir": str(work), "normalizedPath": str(folder / "character.blend"),
                              "posePath": str(copied), "snapshot": snap, "snapshotHash": key,
                              "selectionId": req["selectionId"]}, rid)
        result.update({"localModelId": model_id, "artifactPath": str(cache / "result.fbx"), "cacheHit": False})
        self.progress(rid, "validating", .9)
        (work / "result.json").write_text(json.dumps(result))
        for child in list(work.iterdir()):
            if child.name not in {"result.json", "result.fbx"}:
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        with self.lock:
            self.check_cancel()
            cache.parent.mkdir(exist_ok=True)
            work.rename(cache)
            self.committed = True
        return result

    def execute(self, req):
        rid = req["requestId"]
        work = None
        try:
            work = Path(tempfile.mkdtemp(prefix=uuid.uuid4().hex + "-", dir=self.staging))
            result = self.perform(req, work)
            with self.lock:
                # A commit already performed under the lock wins over late cancellation.
                if not self.committed:
                    self.check_cancel()
                self.active = None
                self.event(rid, "result", result=result)
        except LocalError as exc:
            with self.lock:
                self.active = None
                self.event(rid, "error", error=exc.payload())
        except Exception:
            with self.lock:
                self.active = None
                self.event(rid, "error", error=LocalError("IO_ERROR", "Local storage operation failed").payload())
        finally:
            if work is not None and work.exists():
                shutil.rmtree(work)

    def close(self):
        self.cancelled.set()
        self.pool.shutdown(wait=True)
        self._storage_lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    args = parser.parse_args()
    if not args.storage.is_absolute() or not args.blender.is_absolute():
        parser.error("storage and runtime paths must be absolute")
    stdout_lock = threading.Lock()
    def emit(payload):
        with stdout_lock:
            print(json.dumps(payload, allow_nan=False), flush=True)
    host = Host(args.storage, args.blender, emit)
    try:
        while line := sys.stdin.buffer.readline(LIMITS["requestBytes"] + 1):
            if len(line) > LIMITS["requestBytes"]:
                host.event(None, "error", error=LocalError("INVALID_REQUEST", "Request line exceeds the limit").payload())
                break
            try:
                raw = json.loads(line)
            except (ValueError, UnicodeError, RecursionError):
                host.event(None, "error", error=LocalError("INVALID_REQUEST", "Malformed JSON request").payload())
                continue
            host.submit(raw)
    finally:
        host.close()


if __name__ == "__main__":
    main()
