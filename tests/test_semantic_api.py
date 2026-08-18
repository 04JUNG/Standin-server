"""API/product wrapper regressions for semantic search."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import threading

from fastapi import HTTPException, Response


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import api.app as api_app  # noqa: E402
from api.models import SemanticSearchRequest  # noqa: E402
from src.config import CFG  # noqa: E402
from src.semantic_service import SemanticBusyError, SemanticSearchService  # noqa: E402


class FakeSemanticRuntime:
    def __init__(self) -> None:
        self.calls = 0
        self.units = {"pose:sample": {}}
        self.manifest = {
            "semantic_build_id": "sha256:test-build",
            "semantic_db_schema_version": 2,
            "counts": {"pose_members": 2},
            "embedding": {
                "embedding_version": "test-e5",
                "model_id": "test/model",
                "revision": "test-revision",
            },
        }

    def search(self, query: str, *, top_k: int) -> dict:
        self.calls += 1
        return {
            "query": query,
            "status": "success",
            "exact_match_status": "exact",
            "semantic_build_id": self.manifest["semantic_build_id"],
            "parsed_query": {
                "intent": "observable_constraints",
                "constraint_ids": ["arms_wide"],
            },
            "match_source": "semantic_user",
            "refine_allowed": False,
            "matching_pose_members": 2,
            "matching_semantic_units": 1,
            "unknown_pose_members": 0,
            "results": [
                {
                    "semantic_unit_id": "pose:sample",
                    "pose_id": "sample",
                    "variant_kind": "original",
                    "source_clip_id": "local:sample",
                    "score": 0.02,
                    "constraint_margin": 0.3,
                    "constraint_results": [
                        {
                            "constraint_id": "arms_wide",
                            "state": "exact",
                            "margin": 0.3,
                            "missing_measurements": [],
                        }
                    ],
                    "evidence_state": "observed",
                    "exact_pose_claim": True,
                    "side_resolved": False,
                    "matched_constraints": ["arms_wide"],
                    "unknown_constraints": [],
                    "best_text_document": {
                        "document_id": "sample:posecode:ko",
                        "document_type": "posecode_render",
                        "text": "양팔을 넓게 벌림",
                        "evidence_state": "observed",
                        "candidate_only": False,
                    },
                    "match_source": "semantic_user",
                    "refine_allowed": False,
                }
            ][:top_k],
        }


def _service(runtime=None, **overrides) -> SemanticSearchService:
    return SemanticSearchService(
        runtime or FakeSemanticRuntime(),
        max_concurrency=overrides.get("max_concurrency", 1),
        acquire_timeout_seconds=overrides.get("acquire_timeout_seconds", 0.01),
        cache_size=overrides.get("cache_size", 4),
    )


def test_semantic_service_cache_is_build_and_request_aware() -> None:
    runtime = FakeSemanticRuntime()
    service = _service(runtime)

    first = service.search("양팔을 넓게 벌린 자세", top_k=5, view_hint="front")
    second = service.search("  양팔을   넓게 벌린 자세  ", top_k=5, view_hint="front")
    third = service.search("양팔을 넓게 벌린 자세", top_k=1, view_hint="front")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert third["cache_hit"] is False
    assert runtime.calls == 2
    readiness = service.readiness()
    assert readiness["stats"]["requests"] == 3
    assert readiness["stats"]["cache_hits"] == 1
    assert readiness["stats"]["cache_misses"] == 2


def test_semantic_service_rejects_over_capacity_instead_of_queueing_unbounded() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingRuntime(FakeSemanticRuntime):
        def search(self, query: str, *, top_k: int) -> dict:
            entered.set()
            release.wait(timeout=2)
            return super().search(query, top_k=top_k)

    service = _service(BlockingRuntime(), cache_size=0)
    failures: list[Exception] = []

    def first_request() -> None:
        try:
            service.search("첫 요청", top_k=5)
        except Exception as exc:  # pragma: no cover - diagnostic collection
            failures.append(exc)

    thread = threading.Thread(target=first_request)
    thread.start()
    assert entered.wait(timeout=1)
    try:
        try:
            service.search("두 번째 요청", top_k=5)
        except SemanticBusyError:
            pass
        else:
            raise AssertionError("over-capacity semantic request was accepted")
    finally:
        release.set()
        thread.join(timeout=2)
    assert not failures
    assert service.readiness()["stats"]["busy_rejections"] == 1


def test_semantic_endpoint_maps_product_contract_and_never_allows_refine() -> None:
    previous_state = dict(api_app.STATE)
    previous_max = CFG.semantic_top_k_max
    previous_thumbnail_url = api_app.thumbnail_url
    api_app.STATE.clear()
    api_app.STATE["semantic_service"] = _service()
    CFG.semantic_top_k_max = 20
    api_app.thumbnail_url = (
        lambda _data_dir, pose_id, view: f"/pose/{pose_id}/thumbnail?view={view}"
    )
    try:
        response_headers = Response()
        result = api_app.semantic_search(
            SemanticSearchRequest(
                query="양팔을 넓게 벌린 자세", top_k=5, view_hint="side"
            ),
            response=response_headers,
        ).model_dump()
    finally:
        CFG.semantic_top_k_max = previous_max
        api_app.thumbnail_url = previous_thumbnail_url
        api_app.STATE.clear()
        api_app.STATE.update(previous_state)

    assert result["status"] == "success"
    assert result["match_source"] == "semantic_user"
    assert result["refine_allowed"] is False
    assert result["results"][0]["refine_allowed"] is False
    assert result["results"][0]["preview_view"] == "side"
    assert result["results"][0]["bvh_url"] == "/pose/sample/bvh"
    assert "view=side" in result["results"][0]["thumbnail_url"]
    assert "view_hint_applies_to_preview_only" in result["warnings"]
    assert response_headers.headers["X-Standin-Timing-Kind"] == "semantic-runtime"


def test_semantic_endpoint_is_fail_closed_when_not_ready() -> None:
    previous_state = dict(api_app.STATE)
    api_app.STATE.clear()
    api_app.STATE["semantic"] = api_app._semantic_base_status("disabled")
    try:
        try:
            api_app.semantic_search(SemanticSearchRequest(query="자세"))
        except HTTPException as exc:
            assert exc.status_code == 503
            assert exc.detail["code"] == "semantic_not_ready"
            assert exc.detail["reason"] == "disabled"
        else:
            raise AssertionError("unready semantic endpoint was accepted")
    finally:
        api_app.STATE.clear()
        api_app.STATE.update(previous_state)


def test_semantic_endpoint_maps_capacity_rejection_to_retryable_503() -> None:
    previous_state = dict(api_app.STATE)
    service = _service()

    def reject(*_args, **_kwargs):
        raise SemanticBusyError("semantic search concurrency limit reached")

    service.search = reject  # type: ignore[method-assign]
    api_app.STATE.clear()
    api_app.STATE["semantic_service"] = service
    try:
        try:
            api_app.semantic_search(SemanticSearchRequest(query="양팔을 벌린 자세"))
        except HTTPException as exc:
            assert exc.status_code == 503
            assert exc.detail["code"] == "semantic_busy"
            assert exc.detail["retryable"] is True
            assert exc.headers == {"Retry-After": "1"}
        else:
            raise AssertionError("semantic capacity rejection was not mapped")
    finally:
        api_app.STATE.clear()
        api_app.STATE.update(previous_state)


def test_optional_semantic_failure_does_not_lower_geometry_health() -> None:
    previous_state = dict(api_app.STATE)
    previous_required = CFG.semantic_required
    api_app.STATE.clear()
    api_app.STATE.update(
        {
            "pipeline": object(),
            "pose_count": 1,
            "semantic": api_app._semantic_base_status("startup_failed:FileNotFoundError"),
        }
    )
    CFG.semantic_required = False
    try:
        result = api_app.healthz()
    finally:
        CFG.semantic_required = previous_required
        api_app.STATE.clear()
        api_app.STATE.update(previous_state)

    assert isinstance(result, dict)
    assert result["ok"] is True
    assert result["semantic"]["ready"] is False


def test_required_semantic_failure_lowers_readiness() -> None:
    previous_state = dict(api_app.STATE)
    previous_required = CFG.semantic_required
    api_app.STATE.clear()
    api_app.STATE.update(
        {
            "pipeline": object(),
            "pose_count": 1,
            "semantic": api_app._semantic_base_status("startup_failed:RuntimeError"),
        }
    )
    CFG.semantic_required = True
    try:
        result = api_app.healthz()
    finally:
        CFG.semantic_required = previous_required
        api_app.STATE.clear()
        api_app.STATE.update(previous_state)

    assert isinstance(result, Response)
    assert result.status_code == 503
    payload = json.loads(result.body)
    assert payload["ok"] is False
    assert payload["semantic"]["ready"] is False


def test_production_rejects_unpromoted_semantic_build_at_startup() -> None:
    previous_state = dict(api_app.STATE)
    previous = {
        "app_env": CFG.app_env,
        "semantic_enabled": CFG.semantic_enabled,
        "semantic_required": CFG.semantic_required,
        "semantic_build_dir": CFG.semantic_build_dir,
    }
    previous_loader = api_app.load_semantic_service
    CFG.app_env = "production"
    CFG.semantic_enabled = True
    CFG.semantic_required = True
    CFG.semantic_build_dir = "/explicit/immutable/build"
    api_app.load_semantic_service = lambda **_kwargs: _service()  # type: ignore[assignment]
    api_app.STATE.clear()
    try:
        try:
            api_app._load_semantic_at_startup()
        except api_app.StartupError:
            pass
        else:
            raise AssertionError("unpromoted semantic build started in production")
        assert api_app.STATE["semantic"]["ready"] is False
        assert api_app.STATE["semantic"]["reason"] == "startup_failed:RuntimeError"
    finally:
        for key, value in previous.items():
            setattr(CFG, key, value)
        api_app.load_semantic_service = previous_loader
        api_app.STATE.clear()
        api_app.STATE.update(previous_state)


def test_openapi_exposes_semantic_request_and_response_contract() -> None:
    schema = api_app.app.openapi()
    operation = schema["paths"]["/semantic-search"]["post"]

    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SemanticSearchRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SemanticSearchResponse"
    }


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
