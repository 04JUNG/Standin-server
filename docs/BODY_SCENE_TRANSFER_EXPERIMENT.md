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
`032a2442c` (not authored by this experiment). It synchronizes shared runtime: binary v2,
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

Compare OFF/ON on the same synchronized code, model/source hashes, controls,
prepared setting, v1/v2 encoding and host; fresh output namespaces, sequential
Blender processes. Record failures/recovery alongside success, saved I/O counts,
inclusive body/file-ready wall times and peak RSS. Final files are independently
reloaded and compared for positions, bones/frames, weights, topology and normals.
Required coverage includes male/female base, fat and large custom, ob_015 and a
different approved pose, original/preset hands, scopes and skin/outline contracts.
Binary v2 / prepared-neutral / warm cache gains are excluded from the I/O claim.

No deployment, release, protected branch update or merge is authorized.
