"""Offline executable scene-save-v1 contract scenarios, never app certification.

The fixture driver models the existing BFF/client boundary. Inject a driver
implementing the same methods to collect installed-app observations later.
HTTP cancellation does not claim to stop remote work. Preview failure does not
invalidate already prepared bytes. Actual native saves require disk readback.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from converter.scene_contract import FBX_MAGIC, MAX_PERSONS
from .util import hash_json, sha256_bytes, sha256_file

CONTRACT = "scene-save-v1"
REVIEW_VERSION = "scene-review-v1"
MODES = ("original-selection", "auto-correct-v4")
SOURCE_BINDINGS = {"bff": "96475c9e394976668f9f6ba9ea00268e97775813",
                   "client": "c26572fb4690f132146041321f94c5ead12e86d8"}


class SceneContractError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def verify_scene_download(data, headers, *, person_count, mode, reviewed_sha256=None):
    """Equivalent client header/SHA/binary-magic guard, separate from mesh QA."""
    expected = {"X-Standin-Scene-Contract": CONTRACT, "X-Standin-Scene-Save-Mode": mode,
                "X-Standin-Person-Count": str(person_count),
                "X-Standin-Scene-Version": "scene-refinement-v4" if mode == "auto-correct-v4" else "rough-layout-v1"}
    if any(headers.get(key) != value for key, value in expected.items()):
        raise SceneContractError("SCENE_INTEGRITY")
    digest = sha256_bytes(data)
    if (headers.get("X-Standin-Artifact-SHA256") != digest or not data.startswith(FBX_MAGIC)
            or reviewed_sha256 is not None and reviewed_sha256 != digest):
        raise SceneContractError("SCENE_INTEGRITY")
    return {"sha256": digest, "bytes": len(data), "verified": True}


@dataclass(frozen=True)
class FixtureReceipt:
    binding: str
    sha256: str
    artifact: bytes
    expires: float
    owner: str
    job: str
    review_version: str = REVIEW_VERSION


class OfflineSceneDriver:
    """Synthetic protocol driver. A PASS here is a contract-test assertion only."""
    execution_kind = "fixture"

    def __init__(self, *, mode, person_count, output_dir):
        self.mode, self.person_count = mode, person_count
        self.output = Path(output_dir)
        self.owner, self.job, self.job_revision = "fixture-installation", "fixture-job", 1
        self.settings = {"selection": 1, "body": 1, "hand": 1, "scope": "full", "included": person_count}
        self.deleted, self.http_cancelled, self.preview_state = False, False, "pending"
        self.clock, self.events, self.saved = 0.0, [], []
        self.bytes_ready = False

    def binding(self):
        return hash_json([self.owner, self.job, self.job_revision, self.settings, self.person_count])

    def prepare(self):
        if not 1 <= self.person_count <= MAX_PERSONS:
            raise SceneContractError("INVALID_INPUT")
        if self.mode not in MODES:
            raise SceneContractError("SCENE_CONTRACT_UNSUPPORTED")
        if self.deleted:
            raise SceneContractError("SCENE_OBSERVATION_MISSING")
        # Valid binary prefix, synthetic payload: not a real Blender FBX.
        artifact = FBX_MAGIC + self.binding().encode("ascii")
        self.bytes_ready = True
        self.events.append({"event": "bytes_ready", "binding_sha256": self.binding(), "time": self.clock})
        return FixtureReceipt(self.binding(), sha256_bytes(artifact), artifact, self.clock + 60, self.owner, self.job)

    def change(self, field):
        if field == "scope":
            self.settings[field] = "half"
        elif field == "included":
            self.settings[field] = max(1, self.person_count - 1)
        elif field == "job":
            self.job_revision += 1
        else:
            self.settings[field] += 1
        self.events.append({"event": "settings_changed", "field": field, "time": self.clock})

    def preview(self, *, fail=False, delay=0):
        self.clock += delay
        self.preview_state = "failed" if fail else "ready"
        self.events.append({"event": "preview_" + self.preview_state, "time": self.clock})

    def accept_response(self, receipt):
        accepted = receipt.binding == self.binding() and not self.http_cancelled and not self.deleted
        self.events.append({"event": "response_accepted" if accepted else "response_discarded", "time": self.clock})
        return accepted

    def cancel_http(self):
        self.http_cancelled = True
        self.events.append({"event": "http_cancelled", "server_operation_stopped": False, "time": self.clock})

    def download(self, receipt):
        if self.http_cancelled:
            raise SceneContractError("HTTP_CANCELLED")
        if self.deleted:
            raise SceneContractError("SCENE_OBSERVATION_MISSING")
        if receipt.owner != self.owner or receipt.job != self.job:
            raise SceneContractError("SCENE_OBSERVATION_MISSING")
        if self.mode == "auto-correct-v4" and receipt.expires <= self.clock:
            raise SceneContractError("SCENE_REVIEW_EXPIRED")
        if receipt.binding != self.binding():
            raise SceneContractError("SCENE_SOURCE_CHANGED")
        if self.mode == "auto-correct-v4" and receipt.review_version != REVIEW_VERSION:
            raise SceneContractError("SCENE_REVIEW_REQUIRED")
        headers = {"X-Standin-Scene-Contract": CONTRACT, "X-Standin-Scene-Save-Mode": self.mode,
                   "X-Standin-Person-Count": str(self.person_count), "X-Standin-Artifact-SHA256": receipt.sha256,
                   "X-Standin-Scene-Version": "scene-refinement-v4" if self.mode == "auto-correct-v4" else "rough-layout-v1"}
        checked = verify_scene_download(receipt.artifact, headers, person_count=self.person_count,
                                        mode=self.mode, reviewed_sha256=receipt.sha256 if self.mode == "auto-correct-v4" else None)
        self.events.append({"event": "download_verified", "artifact_sha256": checked["sha256"], "time": self.clock})
        return receipt.artifact

    def native_save(self, data, *, cancel=False, write_fail=False):
        if cancel:
            self.events.append({"event": "native_cancelled", "time": self.clock})
            return {"state": "cancelled", "readback_verified": False}
        if write_fail:
            self.events.append({"event": "native_write_failed", "time": self.clock})
            return {"state": "write_failed", "readback_verified": False}
        self.output.mkdir(parents=True, exist_ok=True)
        index = 0
        while True:
            path = self.output / ("standin_scene.fbx" if index == 0 else f"standin_scene ({index}).fbx")
            try:
                with path.open("xb") as stream:
                    stream.write(data)
                break
            except FileExistsError:
                index += 1
        digest = sha256_file(path)
        result = {"state": "saved", "artifact_sha256": sha256_bytes(data), "saved_sha256": digest,
                  "bytes": path.stat().st_size, "readback_verified": digest == sha256_bytes(data),
                  "execution_kind": self.execution_kind, "native_adapter": "python_fixture_unique_write"}
        self.saved.append((path, digest))
        self.events.append({"event": "native_saved", "artifact_sha256": digest, "time": self.clock})
        return result


def scenario_matrix():
    cases = [{"scenario_id": f"{mode}:{n}-persons", "mode": mode, "person_count": n,
              "action": "save" if n in (1, 2, 3, 8) else "refuse", "expected_code": "INVALID_INPUT" if n in (0, 9) else None}
             for mode in MODES for n in (0, 1, 2, 3, 8, 9)]
    actions = ["preview_pending", "preview_delayed", "preview_failed", "selection", "body", "hand", "scope", "included",
               "job", "expired", "deleted", "other_owner", "other_job", "http_cancel", "late_response", "native_cancel",
               "native_failure", "same_artifact", "after_save_hand_change"]
    for mode in MODES:
        for action in actions:
            cases.append({"scenario_id": f"{mode}:{action}", "mode": mode, "person_count": 3, "action": action})
    return cases


def run_save_scenarios(*, output_dir=None, driver_factory=OfflineSceneDriver):
    """Run every preregistered offline scenario, retaining fail/error rows."""
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="standin-save-contract-") as directory:
            return run_save_scenarios(output_dir=directory, driver_factory=driver_factory)
    rows = []
    for case in scenario_matrix():
        driver = driver_factory(mode=case["mode"], person_count=case["person_count"],
                                output_dir=Path(output_dir) / hash_json(case))
        action, observed, expected = case["action"], None, case.get("expected_code")
        assertions = []
        try:
            receipt = driver.prepare()
            if action in ("selection", "body", "hand", "scope", "included", "job"):
                driver.change(action)
                expected = "SCENE_SOURCE_CHANGED"
            elif action == "expired":
                driver.clock = receipt.expires
                expected = "SCENE_REVIEW_EXPIRED" if case["mode"] == "auto-correct-v4" else None
            elif action == "deleted":
                driver.deleted = True
                expected = "SCENE_OBSERVATION_MISSING"
            elif action in ("other_owner", "other_job"):
                setattr(driver, "owner" if action == "other_owner" else "job", "other-fixture")
                expected = "SCENE_OBSERVATION_MISSING"
            elif action == "http_cancel":
                driver.cancel_http()
                expected = "HTTP_CANCELLED"
            elif action == "late_response":
                driver.change("hand")
                newer = driver.prepare()
                assertions += [driver.accept_response(newer), not driver.accept_response(receipt)]
                receipt = newer
            elif action == "preview_delayed":
                driver.preview(delay=9)
            elif action == "preview_failed":
                driver.preview(fail=True)
            data = driver.download(receipt)
            saved = driver.native_save(data, cancel=action == "native_cancel", write_fail=action == "native_failure")
            if action == "native_cancel":
                assertions += [saved["state"] == "cancelled", not driver.saved]
            elif action == "native_failure":
                assertions += [saved["state"] == "write_failed", not driver.saved]
            else:
                assertions += [saved["state"] == "saved", saved["readback_verified"]]
            if action == "same_artifact":
                again = driver.native_save(driver.download(receipt))
                assertions += [again["saved_sha256"] == saved["saved_sha256"], len(driver.saved) == 2,
                               driver.saved[0][0] != driver.saved[1][0]]
            elif action == "after_save_hand_change":
                path, digest = driver.saved[0]
                driver.change("hand")
                newer = driver.prepare()
                again = driver.native_save(driver.download(newer))
                assertions += [sha256_file(path) == digest, again["saved_sha256"] != digest]
            if action in ("preview_pending", "preview_failed", "preview_delayed"):
                assertions += [driver.bytes_ready, saved["state"] == "saved"]
        except SceneContractError as error:
            observed = error.code
        except Exception as error:
            observed = "driver_error:" + type(error).__name__
        passed = observed == expected and all(assertions)
        rows.append({**case, "state": "pass" if passed else "fail", "observed_code": observed,
                     "expected_code": expected, "events": driver.events,
                     "execution_kind": driver.execution_kind, "product_quality": "unknown",
                     "server_operation_stopped": False if action == "http_cancel" else None})
    return {"schema": "evaluation-extension-v1", "contract": CONTRACT, "source_bindings": SOURCE_BINDINGS,
            "execution_kind": "fixture" if driver_factory is OfflineSceneDriver else "adapter",
            "decision": "INCONCLUSIVE", "contract_tests": {"denominator": len(rows),
                "pass": sum(r["state"] == "pass" for r in rows), "fail": sum(r["state"] == "fail" for r in rows)},
            "product_quality": "unknown", "scenarios": rows, "operational_deployment": False}
