# Body scene transfer experiment (2026-11-13)

Owner: `codex/body-scene-transfer-experiment`, worktree
`/Users/dowon/.codex/worktrees/5a58/Standin-server`.
Start: fetched develop `e9de78d6c7bd99ed3e918a98046e0783d3ae144d`;
reference main `f31558c305c8bb4c1687ab8fe4e15f001ca26f24`.

## I/O boundaries before implementation

| Boundary | Classification | Decision |
| --- | --- | --- |
| Approved author source/hash/controller load | source recovery and provenance | retain |
| Prepared neutral verification, or generated neutral FBX/rest-frame import | coordinate conversion and integrity | retain |
| Neutral import into foot-plant and airborne reference | independent solver/reference inputs | retain initially |
| Exact v3.2.4 parent FBX export/hash and rollback copy | exact-byte recovery | retain |
| Airborne parent reload and solver export | solver canonicalization and recovery | retain initially |
| Solver FBX import into source body | canonical solver input | retain |
| Source body export then import only to obtain deformation matrices | transfer to author, with no separate saved-file gate here | bounded memory binding proxy, then explicit matrices |
| Corrected author export/manifest roundtrip, finger saved-FBX check | geometry/skin/hand loss detection | retain |
| Framing, material typed FBX patch, skin/outline, final independent verifier | delivery and compatibility authority | retain |

The observed nine loader parses / three retarget exports are body_prepare
observations, not total calls or a removal target. Profiling wrappers overlap.
Inclusive worker/body/solver times must not be summed.

## Ownership and prerequisites

The separate **main 공통 수정 develop 동기화** session owns prerequisite
runtime commit `032a2442c`, measured integration head `a770d0cde`, final head
`5e6a63fde` (additional documentation only) in
[PR #237](https://github.com/04JUNG/Standin-server/pull/237) (not authored by this
experiment). It synchronizes shared runtime: binary v2,
metadata/health caching, capacity/error recovery, reviewed ob_083 source frames,
their required modules, image COPY entries and tests. No environment values,
asset pins, deployment workflow, infra, authentication limits or version changes.
The reviewed shape_bake dependency necessarily changes its prepared identity:
develop-built old neutral bundles become invalid and safely regenerate; rebuilding
against this prerequisite is required. The experiment itself changes none of the
prepared-key files. Its earlier local prerequisite `6f9d6c049` is not in the
experiment's final ancestry and has no PR. No checksum bypass or frozen solver edit.

The experiment owns a new scene adapter, worker opt-in connection, explicit image
packaging, related contract/Blender verification and this document. It does not
edit prepare_preset_bodies, package/add-prepared/release helpers, neutral bundles,
or the preset session's docs. PR #235's changed files have no overlap. BFF and
client repositories are read-only references. Existing QA ports and result paths
are untouched. All new outputs use `standin-scene-transfer-20261113-*`.

## Safety and measurement plan

`CONVERTER_EXPERIMENT_BODY_SCENE_TRANSFER=1` opts in; default OFF. ON keeps neutral
prep and solver unchanged. Source body uses the same `apply_source_body`, including
reviewed frames. Before the author stage it exports a binding proxy into a bounded
2 MiB BytesIO using the pinned FBX codec. Object transforms, armature modifier and
bind clusters remain, while the body surface is replaced temporarily with a
three-vertex proxy. The next stage consumes only rig frames, not this proxy surface.
Module-local codec `open` delegates intercept one private sentinel only, restored
in `finally`; original mesh data is restored before scene reset. The sentinel is
never written to disk. This retains codec bone-axis/bind normalization, then passes
a typed scene result within a single Blender job and runs the pinned author's
correction/export/roundtrip unchanged. Unsupported exact-parent jobs use legacy.
Exceptions, cooperative deadline expiry or scene invariants cause legacy retry
from original pinned inputs, never publication of an unchecked candidate.
The existing parent child-process timeout remains the hard execution limit. The
cooperative experimental budget defaults to 120 seconds and is checked between
expensive stages; it is not a preemptive solver timeout. Trace status, recovery
duration and exception class (no private error text) are written to a private
sidecar and included in successful reports.

Two rejected prototypes are preserved in private probe evidence: direct matrices
(max normal vector error 0.00176735, matrix error 2.4483e-5) and an armature-only
memory codec (0.00149425, 4.8161e-5) failed existing tolerances. Keeping binding
clusters with a proxy surface produced exactly zero position/matrix/weight/normal
error for the initial female ob_015 check. No threshold was changed.

Codec call counts do not decrease: a full disk FBX write/read is replaced by a
small memory rig write/read. Reports distinguish total, memory, disk and full-mesh
calls/bytes. Do not describe this as eliminating every intermediate FBX codec.

The adapter has its own qualification hashes for shape orchestration, source-body
code and reviewed-frame data. An upstream quality change uses legacy until the
bridge is re-qualified. These are new adapter guards; the original frozen solver
manifest/checksums and prepared identity code are unchanged. Encoding errors are
re-raised outside Blender's operator boundary after exporter cleanup, then legacy
is retried. The final worker's traceback/error sentinel check is unchanged; an
actual injected codec failure was accepted only after legacy completed and final
FBX verification passed. Memory overflow is checked before BytesIO allocation.

Compare OFF/ON on the same synchronized code, model/source hashes, controls,
prepared setting, v1/v2 encoding and host; fresh output namespaces, sequential
Blender processes. Record failures/recovery alongside success, saved I/O counts,
inclusive body/file-ready wall times and peak RSS. Final files are independently
reloaded and compared for positions, bones/frames, weights, topology and normals.
Required coverage includes male/female base, fat and large custom, ob_015 and a
different approved pose, original/preset hands, scopes and skin/outline contracts.
Binary v2 / prepared-neutral / warm cache gains are excluded from the I/O claim.

No deployment, release, protected branch update or merge is authorized.

## 측정 범위와 해석

`benchmark_body_scene_native.py` uses fixture selections and real native Converter
ASGI routes, approved registry/source assets, fresh private stores, real Blender
prepare/solve and separate-process final export/verify, model GET and file GET.
It opens no listening port and does not run BFF, search, app UI or CSP. The earlier
BFF 34.4/24.5-second observations are not this experiment's OFF baseline.

Every OFF/ON pair fixes model, pose, controls, encoding, code and host. Prepared
neutrals and completed-result caches are disabled on both sides. Each variant
creates a fresh body/solve store. OS file cache is not flushed; order alternates.
The body-base format is v2 on both sides. Primary timing uses v2 state encoding
for all six conditions; an additional female pair explicitly tests v1 state
encoding. No v1→v2 gain is attributed to I/O.

`body_prepare_seconds` in the raw native report times the **prepare_body function
inside Blender**, including run_job, capture, model encoding and snapshot writes.
It excludes Blender startup/shutdown. `body_run_job_seconds` is nested within it.
`file_ready_seconds` covers the ASGI solve/model/export/file sequence and includes
child startups plus independent verification. Never add those durations. RSS is
the maximum individual Blender-child RSS, not total API/BFF/app memory. FBX byte
counts are logical encoded/parsed file sizes, not physical storage-device traffic.

The first native sample overlaps the root's local build window ending
2026-11-13 03:55:14 KST and is retained as contention evidence. The earlier direct
worker matrix is exploratory; its male-large pair overlaps host tests. Final
native measurements start after the build completion notice and final host tests.

Hard parent deadlines/cancellation still terminate the child and reject incomplete
publication; they do not run a second job past the caller deadline. Cooperative
experiment timeout and codec/invariant/qualification failures retry legacy within
the existing parent budget. No process-crash availability improvement is claimed.

## 최종 로컬 결과

| 조건 (v2) | prepare_body 함수 OFF → ON (s) | file-ready OFF → ON (s) | 최대 child RSS OFF → ON (MiB) |
| --- | ---: | ---: | ---: |
| male-base | 28.975 → 27.729 | 40.755 → 38.807 | 1395.4 → 1296.6 |
| male-fat | 34.166 → 35.819 | 44.851 → 47.039 | 1422.6 → 1317.0 |
| male-large | 34.321 → 36.077 | 45.494 → 48.349 | 1374.6 → 1242.9 |
| female-base | 18.764 → 19.586 | 28.306 → 28.225 | 1199.6 → 1096.0 |
| female-fat | 24.711 → 24.457 | 33.674 → 33.742 | 983.7 → 1133.0 |
| female-large | 21.309 → 20.514 | 30.120 → 29.342 | 1105.0 → 1294.3 |

위 표는 macOS arm64 / Blender 5.2.0 build fbe6228777e7 / 15 logical CPU, quota 없는 호스트의 조건별 fresh OFF/ON 1쌍이다. 부모의 첫 모델/상태 캐시는 공유하지 않았다. 6쌍 전부 최종 재로드 정점 위치·본 행렬·가중치·노멀 차이 **0**, 토폴로지 동일이다. 여성 기본형 v1 추가 쌍과 ob_001 원본 손·선택 손/bust, ob_083도 동일성을 통과했다. FBX SHA는 timestamp/메타데이터 차이로 다르며 기하 차이와 구분한다.

매 body_prepare에서 파일 기반 FBX parse **13→12회**, write **6→5회**이고, ON에는 메모리 binding proxy parse/write가 각각 1회 추가된다. 총 codec 호출 수는 동일하다. loader subset은 11회 중 1회가 메모리로, retarget export subset은 3회 중 1회가 메모리로 바뀐다. 전송용 전체 메시 6.70–9.06 MB의 저장·읽기를 약127 KB binding proxy로 대체했다. author/parent/최종 검증은 유지했다. 이 카운터들은 서로 중첩하므로 더하지 않는다.

시간과 RSS가 개선·악화로 섞였고 조건별 표본은 1쌍이다. **전체 최초 준비 시간 또는 메모리의 안정적인 개선은 입증되지 않았다. 기본 OFF를 유지한다.** BFF/운영/실제 앱 UX 성능이나 CSP 신규 수용으로 환산하지 않는다.

성공뿐 아니라 실제 복구도 검증했다. cooperative timeout은 실패 시도 3.547s + legacy 복구 14.946s, codec 강제 실패는 11.903s + 15.024s였다(각 shape orchestrator 내부 구간). 둘 다 최종 기하 오차0이며 원래 최종 worker 오류/저장/검증 게이트를 통과했다. 복구 실패 상태 전파는 계약 테스트로 확인했다.

남녀 OFF/ON의 flesh-guide와 기존 native outline width2를 순차 적용한 4개 조합도 기존 독립 skin/outline verifier를 모두 통과했다. UV·이미지·typed FBX 값 보존·skin binding·outline displacement 검사를 생략하지 않았다. 실제 CSP UI 재검수 및 Linux 실모델 성능 측정은 수행하지 않았다.

정량/원본 해시/한계: [evidence.json](qa/body-scene-transfer-20261113/evidence.json). 실제 FBX/Blend/보고서는 JSON의 raw_artifact_roots 아래 로컬에 보존하며 Git에 넣지 않았다.

최종 검증: host 933 passed / 기존 optional skip1, smoke54/54, compileall·diff-check 통과. 최신 #235 c54ea9724와 merge-tree clean이며 가상 통합 트리의 Converter + prepared release 계약은 956 passed / skip1이다. #172의 기존 충돌 파일8개는 선행 #237 기준과 동일하고 실험이 추가한 충돌 파일은0이다. #235의 소유 스크립트/번들을 수정하지 않았다. #237 최종 head의 추가 변경은 문서뿐이며 재배치 후 모든 측정 실행 코드/프로브 SHA가 동일함을 확인했다.
