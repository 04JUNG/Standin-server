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


## Early multi-person focus extension (2026-11-06)

After existing crop/rescue/repair and before routing/search, multi-person analyses now publish an optional focus with a validated original-image overlay. The existing skeleton/quality mask is reused; invalid/missing/suspect slots and unverified/out-of-frame joints remain absent. We choose one slot with the most validated body points without changing slots, their ordering, indices or any search decision. If no eligible slot has a validated body point, focus is omitted. Single-person behavior is unchanged.

The public focus ID is a request-local `focus_<random>` observation identity, never a personIndex/candidateId. Only the internal live slot reference is retained to associate observations/mode. Existing semantic evidence is attached only when keyed to that identical slot object, and actual search mode only inside that slot's existing search scope. No matching by output number/nearby person/numeric slot ID is attempted. Unknown association => omit observations/mode. Final sorted people clears focus; late same-slot metadata cannot resurrect it.

New raw offline fixture: `tests/fixtures/analysis_progress/focus-producer-v1.json`. Existing raw/public v1 fixtures are preserved unchanged. BFF focus fixture: `/private/tmp/standin-pose-progress-bff/fixtures/analysis-progress/focus-v1.json`.

Validation: 15 producer progress tests, including an Event-gated actual two-person pipeline that observes extracting focus before routing/search runs; identical final normalized results and unchanged VLM/pose/search call counts; invalid/suspect/empty masks; exact-slot-only metadata; final people clearing. Selected regression suite: 124 passed, 3 existing Gemini cases excluded for missing google-genai in the QA environment. Additional model/pose inference/render/search calls and artificial stage timers were not added.

## Original-VLM scene metadata extension (2026-11-06)

The existing extracting snapshot immediately after the original VLM analyze returns now carries optional root `scene: {shot?, view?, personCount?}`. No extra stage, call or event is added. Values remain cumulative through focus, understanding, search and final sorted people, and are scoped to this request/Job.

Shot/view come only from valid `VLMAnalysis.stated_tags` declarations. The control `shot`/`view` enum fields have defaults and are deliberately not read for scene. Global cut view remains distinct from any person's view/description/focus. Person count additionally requires an explicit original raw `num_people` numeric integer in 0..20 matching the existing parsed VLM count. Missing values, strings, booleans, fractions, clamped/out-of-range values, box-count fallbacks and mismatches are omitted; neither people.length nor focus is a count source. An explicit zero is retained. Reading this one count does not expose raw metadata, boxes or dialogue.

The existing metadata-only consumer test now explicitly permits `analysis_progress.py`; search/refine modules still cannot consume observed person tags. Tests cover enum-default provenance, independent omission of bad fields, original vs normalized count, cumulative/request-local state, global vs person view, and observing scene after VLM but before pose begins. Existing A/B tests preserve final results and VLM/pose/search call counts. Auto-refine remains skipped.

New offline raw fixture: `tests/fixtures/analysis_progress/scene-producer-v1.json`, generated with `python -m scripts.analysis_scene_fixture /tmp/pose-scene-producer.json`. It contains seven original-VLM fixture cases (full/half, bust, face, explicit zero, absent/corrupt labels/count, independent valid fields) and 39 snapshots through the production pipeline. All older fixtures are unchanged. Public BFF scene fixture: `/private/tmp/standin-pose-progress-bff/fixtures/analysis-progress/scene-v1.json`.

Validation: selected producer regression suite 158 passed, 3 existing Gemini adapter cases excluded because google-genai is absent in the local QA environment. BFF full suite 988 passed / 2 existing opt-in skips; progress tests 15 passed. The client production parser accepts the original 78 + new 39 = 117 snapshots; the check script asserts scene deep equality. No added model/render/search, query/permission/postprocess/ranking changes, merge, remote deployment or production analysis occurred.
