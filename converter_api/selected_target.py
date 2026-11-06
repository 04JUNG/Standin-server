"""Opt-in private post-selection target provider. Never installs an HTTP route or activates v3/v4.

Selections are registered by trusted server code, never accepted as client paths
or URLs. All artifacts and in-flight work are scoped to the full engine binding.
The frozen converter still needs intermediate FBXs; measure rather than hide it.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from contextvars import ContextVar
from dataclasses import dataclass, replace
import json
import math
import re
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from converter.selected_snapshot import VERSION, encode, revision, sha
from converter.framing import validate_scope
from converter.hand_library import normalize_request, load_library
from converter.scene_contract import FBX_MAGIC, MAX_FBX_BYTES
from converter_api.runner import (BlenderUnavailableError, ConversionTimeoutError,
                                 WorkerIntegrityError, BlenderRunner, RunnerSettings)


class TargetUnsupported(ValueError):
    pass


_private_body_scope = ContextVar("selected_private_body_scope", default=None)


class PrivateBodyCache:
    """Existing verified pre-hand FBX cache, partitioned by private owner/job/person.

    No mutable bpy scene is retained. Restore copies bytes to a private job and
    the unchanged converter validates body lineage and every final hand output.
    Shape-backed .blend reuse is deliberately left to the existing interactive
    adapter; this cache never strips shape correctives to gain a hit.
    """
    def __init__(self, max_bytes, ttl):
        self.max_bytes, self.ttl = max_bytes, ttl
        self._lock, self._entries, self._bytes = threading.Lock(), OrderedDict(), 0

    def _key(self, identity):
        scope = _private_body_scope.get()
        return sha(encode([scope, revision(), identity])) if scope else None

    def _remove(self, key):
        value = self._entries.pop(key)
        self._bytes -= len(value.artifact)+len(value.metadata_json)

    def get(self, identity):
        from converter.retarget_cache import validate_base
        key = self._key(identity)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires <= time.monotonic():
                self._remove(key)
                return None
            validate_base(entry.artifact, json.loads(entry.metadata_json), identity)
            self._entries.move_to_end(key)
            return entry

    def put(self, identity, data, metadata_json):
        from converter.retarget_cache import validate_base, Entry
        key = self._key(identity)
        size = len(data)+len(metadata_json)
        if key is None or size > self.max_bytes:
            return False
        validate_base(data, json.loads(metadata_json), identity)
        with self._lock:
            if key in self._entries:
                self._remove(key)
            while self._entries and self._bytes+size > self.max_bytes:
                self._remove(next(iter(self._entries)))
            self._entries[key] = Entry(bytes(data), bytes(metadata_json), time.monotonic()+self.ttl)
            self._bytes += size
        return True


class SelectedBlenderRunner(BlenderRunner):
    """Dedicated child entrypoint; keeps one shared process/admission budget."""
    def __init__(self, settings=None):
        settings = settings or replace(RunnerSettings.from_env(), retarget_cache_bytes=32*1024*1024)
        environment = {**(settings.process_env or {}), "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        super().__init__(replace(settings, process_env=environment,
            worker_path=Path(__file__).resolve().parents[1] / "converter/selected_convert_worker.py"))
        self.retarget_cache = PrivateBodyCache(settings.retarget_cache_bytes, settings.retarget_cache_ttl_seconds)

    def _command(self, job_path):
        command = super()._command(job_path)
        return [command[0], "--factory-startup", "--disable-autoexec", "--threads", "1", *command[1:]]


@dataclass(frozen=True)
class TargetSelection:
    """Immutable server-owned cohort: model + shape + hands + range + appearance."""
    character_id: str
    character_path: Path
    character_sha256: str
    rig_profile: str = "mixamo"
    shape_recipe_json: bytes = b"null"
    hands_json: bytes = b"null"
    output_scope: str = "full"
    appearance_id: str = "default"

    def identity(self):
        recipe = json.loads(self.shape_recipe_json)
        hands = normalize_request(json.loads(self.hands_json))
        validate_scope(self.output_scope)
        if self.rig_profile != "mixamo" or self.appearance_id != "default":
            raise TargetUnsupported("target profile or appearance adapter unavailable")
        return {"character_id": self.character_id, "character_sha256": self.character_sha256,
                "rig_profile": self.rig_profile, "shape_sha256": sha(encode(recipe)),
                "hands": hands, "hand_library_sha256": load_library().sha256,
                "output_scope": self.output_scope, "appearance_id": self.appearance_id}

    def verify(self):
        identity = self.identity()
        if (self.character_path.is_symlink() or not self.character_path.is_file()
                or sha(self.character_path.read_bytes()) != self.character_sha256):
            raise WorkerIntegrityError("registered target asset changed")
        recipe = json.loads(self.shape_recipe_json)
        if recipe is not None:
            from converter.shape_spec import validate_source
            validate_source(recipe["rules"], {key: recipe[key] for key in (
                "blend", "controller", "blend_sha256", "controller_sha256")})
            if recipe["asset_sha256"] != self.character_sha256:
                raise WorkerIntegrityError("registered target shape mismatch")
        return identity


@dataclass(frozen=True)
class PreparedTarget:
    artifact: bytes
    preview: bytes
    receipt_json: bytes

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


def inspect_target(runner, artifact, binding):
    """Use the SAME global process slots and cancellation/kill semantics as export."""
    if not artifact.startswith(FBX_MAGIC) or len(artifact) > MAX_FBX_BYTES:
        raise WorkerIntegrityError("invalid candidate final FBX")
    runner.inspect_blender()
    with runner._process_slots, tempfile.TemporaryDirectory(
            prefix="standin-candidate-check-", dir=runner.settings.temp_root) as directory:
        root = Path(directory).resolve()
        (root / "final.fbx").write_bytes(artifact)
        job = {"artifact_sha256": sha(artifact), "binding": binding, "revision": revision()}
        job_bytes = encode(job)
        path = root / "job.json"
        path.write_bytes(job_bytes)
        command = [runner.settings.blender_binary, "--background", "--factory-startup",
                   "--disable-autoexec", "--threads", "1",
                   "--python-exit-code", "1", "--python",
                   str(Path(__file__).resolve().parents[1] / "converter/selected_snapshot_worker.py"),
                   "--", "--job", str(path)]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True, env=runner._environment())
        except OSError as exc:
            raise BlenderUnavailableError("candidate inspection could not start") from exc
        try:
            runner._communicate(process, runner.settings.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            runner._terminate_process_group(process, runner.settings.terminate_grace_seconds)
            raise ConversionTimeoutError("candidate inspection timed out") from exc
        report_path, preview_path = root / "report.json", root / "preview.glb"
        if process.returncode or report_path.is_symlink() or preview_path.is_symlink():
            raise WorkerIntegrityError("candidate inspection worker failed")
        try:
            from converter.preview_model import MAX_MODEL_BYTES
            if preview_path.stat().st_size > MAX_MODEL_BYTES or report_path.stat().st_size > 4*1024*1024:
                raise ValueError("candidate output too large")
            report = json.loads(report_path.read_bytes())
            preview = preview_path.read_bytes()
            if (report.get("ok") is not True or report.get("job_sha256") != sha(job_bytes)
                    or report.get("revision") != job["revision"]
                    or sha((root / "final.fbx").read_bytes()) != sha(artifact)):
                raise ValueError("candidate inspection lineage changed")
            validate_prepared(PreparedTarget(artifact, preview, encode(report)), binding)
            return preview, report
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise WorkerIntegrityError("candidate inspection integrity mismatch") from exc


def validate_prepared(value, binding):
    from converter.final_artifact import valid_receipt
    from converter.framed_model import metadata
    from converter.hand_contact import validate_screen
    report = value.receipt
    error = report.get("correspondence", {}).get("max_coordinate_error_m")
    if (report.get("version") != VERSION or report.get("binding") != binding
            or report.get("artifact_sha256") != sha(value.artifact)
            or report.get("preview_sha256") != sha(value.preview)
            or report.get("preview_size") != len(value.preview)
            or report.get("baseline_restored") is not True
            or not isinstance(report.get("baseline_sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", report["baseline_sha256"])
            or report.get("baseline_sha256") != report.get("restored_sha256")
            or report.get("correspondence", {}).get("surface_verified") is not True
            or type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 0.0001
            or not valid_receipt(report.get("structure"), sha(value.artifact), binding["output_scope"])):
        raise WorkerIntegrityError("prepared target receipt mismatch")
    meta = metadata(value.preview)
    if meta.get("binding") != binding or meta.get("artifact_sha256") != sha(value.artifact):
        raise WorkerIntegrityError("prepared preview lineage mismatch")
    validate_screen(report.get("hands"), artifact_sha256=sha(value.artifact))
    mesh = report.get("mesh", {})
    if (mesh.get("status") not in ("mesh_validated", "mesh_unverified", "validation_blocked")
            or mesh.get("complete_collision_certificate") is not False):
        # No accepted full collision policy exists. Do not promote an injected
        # true certificate, even if some other process supplied one.
        raise WorkerIntegrityError("unaccepted target collision certificate")
    return report


class SelectedTargetProvider:
    def __init__(self, runner, selections, *, max_cache_bytes=64*1024*1024,
                 ttl_seconds=600., inspector=inspect_target, clock=time.monotonic,
                 is_current):
        if (type(max_cache_bytes) is not int or max_cache_bytes < 0
                or not math.isfinite(ttl_seconds) or ttl_seconds <= 0):
            raise ValueError("invalid target cache bounds")
        self.runner, self.selections = runner, dict(selections)
        self.max_bytes, self.ttl, self.clock = max_cache_bytes, ttl_seconds, clock
        self.inspector = inspector
        self.is_current = is_current
        self._lock = threading.Lock()
        self._cache, self._flights, self._bytes = OrderedDict(), {}, 0

    def _context(self, source, binding, target):
        if not isinstance(target, dict) or set(target) != {"selection_id", "revision"}:
            raise TargetUnsupported("expected registered target selection identity")
        selection = self.selections.get(target["selection_id"])
        if selection is None:
            raise TargetUnsupported("unknown target selection")
        identity = selection.verify()  # Also on cache hit/reuse: never trust a path alone.
        if target["revision"] != sha(encode(identity)):
            raise TargetUnsupported("target selection revision changed")
        if (not isinstance(binding, dict) or not all(binding.get(key) for key in (
                "owner_id", "job_id", "person_id", "candidate_id", "candidate_set_revision",
                "context_fingerprint", "selection_revision", "prepared_source_id")) or binding.get("output_sha256") != sha(source)):
            raise ValueError("target source/binding mismatch")
        if binding.get("phase") != "selected_target":
            raise TargetUnsupported("body application requires an authoritative selected-target binding")
        if not self.is_current(binding):
            raise TargetUnsupported("selected target revision is no longer current")
        context = {"engine_binding": json.loads(encode(binding)), "target": identity,
                   "output_scope": selection.output_scope, "revision": revision()}
        return selection, context

    def prepare(self, source_bvh_bytes, binding, target):
        selection, context = self._context(source_bvh_bytes, binding, target)
        key = sha(encode(context))
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                value, expires = entry
                if expires > self.clock():
                    validate_prepared(value, context)
                    self._cache.move_to_end(key)
                    return value
                self._remove(key)
            future = self._flights.get(key)
            leader = future is None
            if leader:
                future = self._flights[key] = Future()
        if not leader:
            # Follower cancellation does not release another candidate's slot.
            from converter_api.work_queue import current_work
            while not future.done():
                control = current_work.get()
                if control:
                    control.check("target_single_flight")
                try:
                    value = future.result(timeout=.05)
                    break
                except TimeoutError:
                    pass
            else:
                value = future.result()
            selection.verify()
            if not self.is_current(binding):
                raise TargetUnsupported("selection changed while awaiting target preparation")
            return value
        try:
            value = self._prepare(source_bvh_bytes, selection, context, key)
            selection.verify()
            if not self.is_current(binding):
                raise TargetUnsupported("selection changed during target preparation")
            validate_prepared(value, context)
            size = len(value.artifact)+len(value.preview)+len(value.receipt_json)
            with self._lock:
                if size <= self.max_bytes:
                    while self._cache and self._bytes+size > self.max_bytes:
                        self._remove(next(iter(self._cache)))
                    self._cache[key] = (value, self.clock()+self.ttl)
                    self._bytes += size
            future.set_result(value)
            return value
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._flights.pop(key, None)

    def _remove(self, key):
        value, _ = self._cache.pop(key)
        self._bytes -= len(value.artifact)+len(value.preview)+len(value.receipt_json)

    def _prepare(self, source, selection, context, key):
        started = self.clock()
        binding = context["engine_binding"]
        scope = sha(encode({k: binding[k] for k in ("owner_id", "job_id", "person_id", "candidate_set_revision")}))
        token = _private_body_scope.set(scope)
        try:
            result = self.runner.convert(bvh_bytes=source, character_path=selection.character_path,
                character_id=selection.character_id, character_sha256=selection.character_sha256,
                conversion_id=key[:32], output_scope=selection.output_scope,
                shape_recipe=json.loads(selection.shape_recipe_json), hands=json.loads(selection.hands_json),
                thumbnail=None, preview_view=None, preview_model=False, camera_rotation=None)
        finally:
            _private_body_scope.reset(token)
        preview, report = self.inspector(self.runner, result.artifact, context)
        report.update(conversion_ms=result.execution_ms, queue_ms=result.queue_wait_ms,
                      total_ms=(self.clock()-started)*1000,
                      preview_revision=sha(encode(context)+sha(result.artifact).encode()+sha(preview).encode()),
                      export_authorized=False,
                      quality_scope="target_structural_checked",
                      target_refine=False, automatic_parts=False, skin_safe=False,
                      scene_joint_validation=False, intermediate_fbx_required=True)
        report["body_cache"] = result.report.get("retarget_reuse", {}).get("mode", "disabled")
        if report["hands"]["status"] == "intersection" or report["mesh"]["hard_violations"]:
            raise WorkerIntegrityError("candidate target geometry screen blocked", report=report)
        return PreparedTarget(bytes(result.artifact), bytes(preview), encode(report))

    def reuse(self, source_bvh_bytes, binding, target):
        """Selection reuse never computes: an expired/missing entry is an error."""
        _, context = self._context(source_bvh_bytes, binding, target)
        key = sha(encode(context))
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry[1] <= self.clock():
                raise KeyError("prepared target expired or missing")
            value = entry[0]
            validate_prepared(value, context)
            return value

    def prepare_or_base(self, source_bvh_bytes, base_bvh_bytes, binding, target):
        """Independent same-condition fallback. A failed base is never ready_base."""
        if binding.get("source_sha256") != sha(base_bvh_bytes):
            raise ValueError("fallback must be the authoritative original source")
        try:
            status = "ready_base" if source_bvh_bytes == base_bvh_bytes else "ready_refined"
            return status, self.prepare(source_bvh_bytes, binding, target)
        except WorkerIntegrityError:
            if source_bvh_bytes == base_bvh_bytes:
                raise
            base_binding = {**binding, "output_sha256": sha(base_bvh_bytes),
                            "fallback_from_sha256": sha(source_bvh_bytes)}
            return "ready_base", self.prepare(base_bvh_bytes, base_binding, target)
