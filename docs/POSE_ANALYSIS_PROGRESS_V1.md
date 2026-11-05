# pose-analysis-progress.v1 — inference producer

Shared contract: `/Users/dowon/.codex/worktrees/pose-search-film-loading/Standin-client/docs/POSE_ANALYSIS_PROGRESS_V1.md`.
BFF implementation/report: `/Users/dowon/.codex/worktrees/eef0/Standin-app-server/docs/POSE_ANALYSIS_PROGRESS_V1.md`.

## Transport and actual milestones

`POST /analyze` retains its JSON response. Only `Accept: application/x-ndjson` opts into newline records `{type:"progress",progress}`, `{type:"result",result}` or `{type:"error",status,detail}`. Same analysis call, same final schema, no retry/fallback request, additional VLM invocation or render. Proxy buffering must be disabled for this content type; response sets `X-Accel-Buffering: no` and `Cache-Control: no-store`.

- analyzing: entering existing pipeline, completed when initial VLM returns.
- extracting: existing detector/pose/crop/rescue/repair work; validated mask after recovery, before routing. Coordinates use original input pixels divided by original width/height, never crop-local pixels; invalid/unverified/suspect joints are null/absent. Face points are null.
- understanding: existing description result joins. Only explicit `body_observations` enum fields leave the producer; no prose/context/prompt/trace. No description call means skipped, not completed.
- searching: mode recorded at actual geometry/semantic submission, including zero-result/error searches. Whole-library parallel searches -> combined, semantic only -> description, observed upper geometry -> upper_body, other geometry -> pose. Policy-blocked/basic-model branches omit mode.
- refining: skipped. Current `/analyze` never invokes selected-candidate `/refine`. Eligibility and generated research/hand repair are not that operation. No running/before/after is manufactured and no new auto-refine is enabled.
- ranking: existing final slot/candidate order. Dynamic candidate IDs are published only by existing API asset/camera binding, using the exact final presentation ID and static thumbnail. Missing thumbnails omit previews. No aligned/poster/lookup/renderer URL is emitted. Ranking completes after final API model construction.

Multi-person crop retries can change left/right output order. Before that order is final, person payloads are withheld (stage milestones still emit). After sorting, snapshot people use the same indices as the final result. Single-person skeleton/observations/mode arrive before search completion. Global completed means a stage ran for at least one person; global skipped only means no person ran it. Completed wins over a skipped parallel branch.

Request-local context and lock serialize cumulative snapshots. One-slot transport queue coalesces intermediate snapshots; terminal events cannot be replaced by late updates. Existing analysis does not wait for presentation animation, transport replay, or downstream progress storage. On disconnect no extra inference is started; already running analysis follows existing lifetime/cancellation behavior.

## Offline verification

```bash
python -m pytest tests/test_analysis_progress.py tests/test_description_prefetch.py tests/test_search_routing.py tests/test_fullbody_routing.py tests/test_analyze_upstream_errors.py -q
python -m scripts.analysis_progress_fixture /tmp/pose-progress-producer.json
```

`tests/fixtures/analysis_progress/producer-v1.json` is explicitly OFFLINE mock-model data captured through the production pipeline. It includes combined/description 1p/2p and mixed legacy pose+description. It proves wiring/call counts, not live model accuracy. `test_analysis_progress.py` tests original-image masks, existing-thumbnail-only policy, request isolation, stage overlap, real milestone delivery before completion, immediate terminal/error behavior and result/call-count equivalence. Current validation environment lacks google-genai; three existing Gemini adapter tests were deselected with `-k 'not gemini'`, and all other selected tests passed. No production analysis, deploy or merge occurred.

For an offline HTTP smoke server (synthetic library/mock backends, no artwork-understanding claim):

```bash
VLM_PROVIDER=mock POSE_BACKEND=mock ROUTING_FULLBODY_ENABLED=0 BODY_MATCHING_MODE=off \
VLM_PERSON_TAGS=0 VLM_HAND_SHAPES=0 POSE_LIBRARY_URI='' POSE_MODEL_URI='' \
DATA_DIR=/tmp/pose-progress-api DB_PATH=/tmp/pose-progress-api/poses.db \
python -m uvicorn api.app:app --host 127.0.0.1 --port 18000
```

Install repository requirements in a local environment first. With no prepared thumbnail files the correct output has no previews; the mock adapter also has no real observation keywords. For actual local model integration use your existing local model/library configuration and this worktree's API; do not point the smoke commands at production.
