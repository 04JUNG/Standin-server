"""Post-selection authority, immutable reuse and independent surface checks."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import struct
import threading
from types import SimpleNamespace

import pytest

from converter.selected_snapshot import VERSION, encode, sha, verify_glb
from converter.preview_model import encode_surface
from converter.scene_contract import FBX_MAGIC
from converter.hand_contact import SCREEN_VERSION
from converter_api.selected_target import (SelectedTargetProvider, TargetSelection,
    TargetUnsupported, PreparedTarget, validate_prepared, PrivateBodyCache, _private_body_scope)
from converter_api.runner import WorkerIntegrityError, ConversionTimeoutError


SOURCE = b"selected prepared source"
POSITIONS = [0., 0., 0., 1., 0., 0., 0., 1., 0.]
INDICES = [0, 1, 2]


def inspected(runner, artifact, binding):
    preview = encode_surface(POSITIONS, [0., 0., 1.]*3, INDICES,
        {"binding": binding, "artifact_sha256": sha(artifact)})
    return preview, {"version": VERSION, "binding": binding,
        "artifact_sha256": sha(artifact), "preview_sha256": sha(preview), "preview_size": len(preview),
        "baseline_restored": True, "baseline_sha256": "a"*64, "restored_sha256": "a"*64,
        "correspondence": verify_glb(preview, POSITIONS, INDICES),
        "structure": {"version": "final-fbx-v1", "artifact_sha256": sha(artifact),
            "scope": binding["output_scope"], "skin_verified": True,
            "bones": 22, "meshes": 1, "vertices": 3, "polygons": 1},
        "mesh": {"status": "mesh_unverified", "hard_violations": [],
                 "complete_collision_certificate": False},
        "hands": {"method": SCREEN_VERSION, "scope": "opposite_hands_only",
                  "status": "clear_screen", "triangle_counts": {"left": 1, "right": 1},
                  "intersection_pairs": 0, "complete_collision_certificate": False,
                  "screened_artifact_sha256": sha(artifact)}}


class Runner:
    def __init__(self):
        self.calls = []
    def convert(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(artifact=FBX_MAGIC+kwargs["bvh_bytes"], execution_ms=2., queue_wait_ms=0., report={})


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "target.fbx"
    path.write_bytes(b"registered-target")
    selection = TargetSelection("target", path, sha(path.read_bytes()))
    target = {"selection_id": "current", "revision": sha(encode(selection.identity()))}
    binding = {"owner_id": "owner", "job_id": "job", "person_id": "person",
        "candidate_id": "candidate", "candidate_set_revision": "set-1",
        "context_fingerprint": "b"*64, "output_sha256": sha(SOURCE), "source_sha256": sha(b"base"),
        "phase": "selected_target", "selection_revision": 1, "prepared_source_id": "source-1"}
    runner = Runner()
    provider = SelectedTargetProvider(runner, {"current": selection}, inspector=inspected,
                                      is_current=lambda value: True)
    return provider, runner, selection, binding, target


def test_reuse_never_solves_renders_or_exports_and_preserves_exact_bytes(setup):
    provider, runner, _, binding, target = setup
    prepared = provider.prepare(SOURCE, binding, target)
    assert provider.prepare(SOURCE, binding, target) is prepared
    assert provider.reuse(SOURCE, binding, target) is prepared
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["thumbnail"] is None and call["preview_view"] is None
    assert call["preview_model"] is False and call["camera_rotation"] is None
    assert prepared.receipt["skin_safe"] is False
    assert prepared.receipt["target_refine"] is False


@pytest.mark.parametrize("change", [{"phase": "pre_top5"}, {"selection_revision": None},
    {"prepared_source_id": None}, {"output_sha256": "f"*64}])
def test_preselection_or_unbound_bytes_never_start_target(setup, change):
    provider, runner, _, binding, target = setup
    with pytest.raises(ValueError):
        provider.prepare(SOURCE, {**binding, **change}, target)
    assert runner.calls == []


@pytest.mark.parametrize("field", ["owner_id", "job_id", "person_id", "candidate_id",
    "candidate_set_revision", "context_fingerprint", "selection_revision"])
def test_private_context_and_revision_never_share_results(setup, field):
    provider, runner, _, binding, target = setup
    provider.prepare(SOURCE, binding, target)
    changed = {**binding, field: 2 if field == "selection_revision" else "changed"}
    with pytest.raises(KeyError):
        provider.reuse(SOURCE, changed, target)
    provider.prepare(SOURCE, changed, target)
    assert len(runner.calls) == 2


def test_corrupt_model_or_recipe_stops_even_cached_selection(setup):
    provider, runner, selection, binding, target = setup
    provider.prepare(SOURCE, binding, target)
    selection.character_path.write_bytes(b"tampered")
    with pytest.raises(WorkerIntegrityError):
        provider.reuse(SOURCE, binding, target)
    assert len(runner.calls) == 1


def test_ttl_eviction_never_recomputes_on_selected_reuse(setup):
    provider, runner, _, binding, target = setup
    provider.clock = lambda: 0
    provider.prepare(SOURCE, binding, target)
    provider.clock = lambda: 601
    with pytest.raises(KeyError):
        provider.reuse(SOURCE, binding, target)
    assert len(runner.calls) == 1


def test_single_flight_for_identical_selected_binding(setup):
    provider, runner, _, binding, target = setup
    started, release = threading.Event(), threading.Event()
    def blocking(*args):
        started.set()
        assert release.wait(5)
        return inspected(*args)
    provider.inspector = blocking
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(provider.prepare, SOURCE, binding, target)
        assert started.wait(5)
        second = pool.submit(provider.prepare, SOURCE, binding, target)
        release.set()
        assert first.result() is second.result()
    assert len(runner.calls) == 1


def test_late_selection_is_discarded_and_not_cached(setup):
    provider, runner, _, binding, target = setup
    current = [True]
    provider.is_current = lambda value: current[0]
    def late(*args):
        current[0] = False
        return inspected(*args)
    provider.inspector = late
    with pytest.raises(TargetUnsupported):
        provider.prepare(SOURCE, binding, target)
    assert not provider._cache and not provider._flights


def test_blocked_refined_falls_back_to_independently_checked_same_conditions(setup):
    provider, runner, _, binding, target = setup
    def screen(*args):
        preview, report = inspected(*args)
        if args[1] == FBX_MAGIC+SOURCE:
            report["mesh"]["hard_violations"] = ["skin_triangle_intersection"]
        return preview, report
    provider.inspector = screen
    status, value = provider.prepare_or_base(SOURCE, b"base", binding, target)
    assert status == "ready_base" and value.artifact == FBX_MAGIC+b"base"
    assert len(runner.calls) == 2
    assert runner.calls[0]["shape_recipe"] == runner.calls[1]["shape_recipe"]
    assert runner.calls[0]["hands"] == runner.calls[1]["hands"]
    with pytest.raises(KeyError):
        provider.reuse(SOURCE, binding, target)


def test_timeout_is_retryable_never_success_or_ready_base(setup):
    provider, runner, _, binding, target = setup
    def timeout(*args):
        raise ConversionTimeoutError("timeout")
    provider.inspector = timeout
    with pytest.raises(ConversionTimeoutError):
        provider.prepare_or_base(SOURCE, b"base", binding, target)
    assert not provider._cache and len(runner.calls) == 1


def test_independent_glb_decoder_detects_triangle_corruption():
    data = bytearray(encode_surface(POSITIONS, [0., 0., 1.]*3, INDICES, {}))
    size = struct.unpack_from("<I", data, 12)[0]
    struct.pack_into("<f", data, 28+size, .2)
    with pytest.raises(ValueError, match="differs"):
        verify_glb(bytes(data), POSITIONS, INDICES)


@pytest.mark.parametrize("change", [{"baseline_restored": False}, {"restored_sha256": "c"*64},
    {"baseline_sha256": None}, {"artifact_sha256": "c"*64}])
def test_receipt_tampering_fails_closed(setup, change):
    provider, _, _, binding, target = setup
    value = provider.prepare(SOURCE, binding, target)
    altered = replace(value, receipt_json=encode({**value.receipt, **change}))
    with pytest.raises(WorkerIntegrityError):
        validate_prepared(altered, value.receipt["binding"])


def test_unaccepted_mesh_certificate_cannot_be_promoted(setup):
    provider, _, _, binding, target = setup
    def forged(*args):
        preview, report = inspected(*args)
        report["mesh"]["complete_collision_certificate"] = True
        return preview, report
    provider.inspector = forged
    with pytest.raises(WorkerIntegrityError):
        provider.prepare(SOURCE, binding, target)


def test_body_cache_reuses_only_same_private_cohort_and_detects_corruption():
    from tests.converter.test_retarget_cache import IDENTITY, DATA, packet
    identity, data, metadata = IDENTITY, DATA, encode(packet())
    cache = PrivateBodyCache(100000, 600.)
    # A caller without an authoritative selected scope never gets a shared base.
    assert cache.put(identity, data, metadata) is False
    token = _private_body_scope.set("owner-job-person-1")
    try:
        assert cache.put(identity, data, metadata)
        entry = cache.get(identity)
        assert entry.artifact == data
        _private_body_scope.set("owner-job-person-2")
        assert cache.get(identity) is None
        _private_body_scope.set("owner-job-person-1")
        assert cache.get(identity) is entry
        cache._entries[cache._key(identity)] = replace(entry, artifact=data+b"corrupt")
        with pytest.raises(ValueError):
            cache.get(identity)
    finally:
        _private_body_scope.reset(token)


def test_new_modules_are_packaged_and_contract_tests_run_in_ci():
    root = Path(__file__).resolve().parents[2]
    docker = (root / "Dockerfile.converter").read_text()
    for name in ("converter/selected_snapshot.py", "converter/selected_snapshot_worker.py",
                 "converter/selected_convert_worker.py", "converter/mesh_validation.py",
                 "converter_api/selected_target.py"):
        assert name in docker
    assert "tests/converter/test_selected_target.py" in (root / ".github/workflows/converter-ci.yml").read_text()


def test_arbitrary_fallback_bytes_are_rejected_before_work(setup):
    provider, runner, _, binding, target = setup
    with pytest.raises(ValueError, match="authoritative original"):
        provider.prepare_or_base(SOURCE, b"unregistered-base", binding, target)
    assert not runner.calls


def test_selected_child_limits_threads_and_disables_embedded_scripts(tmp_path):
    from converter_api.selected_target import SelectedBlenderRunner
    command = SelectedBlenderRunner()._command(tmp_path / "job.json")
    assert "--factory-startup" in command and "--disable-autoexec" in command
    assert command[command.index("--threads")+1] == "1"
    assert "selected_convert_worker.py" in " ".join(command)


def test_unmodified_source_is_never_labelled_refined(setup):
    provider, runner, _, binding, target = setup
    binding = {**binding, "source_sha256": sha(SOURCE)}
    status, value = provider.prepare_or_base(SOURCE, SOURCE, binding, target)
    assert status == "ready_base" and value.artifact == FBX_MAGIC+SOURCE
