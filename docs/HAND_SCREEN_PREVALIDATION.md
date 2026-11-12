# Prevalidated library hand screening

Instead of rebuilding and inspecting the same library pose for every request,
an operator can inspect fixed release assets ahead of time. Requests then check
that the pose, body, hands and actual Converter runtime match that inspection.
This feature is disabled by default and does not deploy or upload assets.

## Binding and failure rules

Each certificate binds the library pose ID, exact BVH bytes, character ID and
FBX SHA256, normalized left/right hand presets, hand-library hash, solver and
screen versions, and a Converter runtime fingerprint. The fingerprint includes
the actual distribution's Python/JSON sources, Blender version/build, OS/CPU
architecture, exact-solver switch and wrist transfer settings.

The catalog is an operator-controlled release file with a mandatory SHA256 pin.
Malformed or tampered catalogs prevent activation. Runtime clients cannot supply
catalog paths or certificates. Requests never publish new entries. A fresh
internal profile request verifies the current Converter and rehashes its actual
character asset once per screening session; there is no cross-request profile TTL.

Only verified library files can use certificates. Their path, symlink and current
BVH SHA256 checks still run first. Missing entries, changed geometry/code/presets,
an old Converter without the profile endpoint, or profile failures use live
screening. Unknown is never a pass. Both clear and intersection results can be
prevalidated; candidate order and rejected candidates are preserved.

Catalog-enabled library selection commits in order, including mixed certificate
hits and live misses. Misses in this mode run live sequentially even if the
configured parallel capacity is greater than one. Generated candidates retain
the existing bounded parallel path. Partial catalog coverage can therefore be
slower than parallel live screening; measure coverage before enabling it.

Generated/custom BVHs always use live checks, even when their bytes equal a
library pose previously checked in the same session. Their session-local cache
is separate from library certificates. Check limits, work/wall budgets, optional
priority, cancellation, presentation and export/refine permissions remain in force.

## Body scope

Certificates apply to one exact base FBX, not a label such as "male" or "female".
The two base characters have independent IDs and file hashes. Both must be
validated separately. Every certificate explicitly has `shape_sha256: null`.
Shape recipes cannot borrow base-body certificates; the current hand-screen
endpoint does not accept shape recipes, and explicitly configured shape requests
remain unavailable rather than receiving a base-body pass.

The existing pipeline searches/screens using its configured fixed character
before attaching body recommendations. This feature preserves that existing
gate's scope. It does not establish collision clearance for a body chosen later,
continuous body morphs, or edited/refined output. Full body-specific screening
requires passing the selected body/shape recipe to a compatible checker after
that choice; it remains separate work. A clear hand receipt certifies only
opposite-hand triangle crossing and grants no actual-mesh/export permission.

## Offline preparation

The caller-owned manifest lists `fixtures: [{id, bvh, sha256}]`; BVH paths are
relative to it. Optional `hand_settings: [{id, hands}]` can list original and
specific presets. Omitting hand settings certifies only the unchanged hands.

```sh
python scripts/prevalidate_hand_screen_library.py \
  --manifest /private/library-manifest.json \
  --converter-url http://127.0.0.1:8001 \
  --character-id standin-master-v2 --character-sha256 "$CHARACTER_SHA256" \
  --output /private/new-hand-catalog.json
```

The builder uses the existing live checker for every combination. It refuses
source hash mismatches, transient/unavailable results, a different body binding
or a profile change during preparation. It never overwrites an existing output.
It prints the catalog's SHA256; configure both `HAND_SKIN_PREVALIDATED_PATH` and
`HAND_SKIN_PREVALIDATED_SHA256` to opt in. A partial pair is rejected. Clearing
both returns to live screening. No catalog or deployment variables were activated
as part of this local preparation.

Prepare certificates on the intended target distribution and platform. Local
Mac certificates intentionally do not apply to Linux/amd64 Converter containers.
Rebuild them after changes to the runtime, character, source pose or hand presets.
Coverage must be checked over the actual library, bodies and hand settings before
activation. This experiment certified only six pose fixtures per body/hand setting.

## Local comparison, 2026-10-30

Real loopback HTTP and Blender 5.2.0 (`fbe6228777e7`) on macOS arm64. Two exact base
FBXs, six fixed BVHs and two hand settings; two crossed rounds per condition.
Each row's values are medians. Offline preparation is excluded from request time
and reported separately. Initial process startup and server-wide concurrency are
not represented by these warm local calls.

| Base body | Hands | Live, one worker | Live, two workers | Prevalidated, one worker |
| --- | --- | ---: | ---: | ---: |
| Male base | Original | 8.580 s | 4.582 s | 5.684 ms |
| Male base | Both relaxed | 9.215 s | 4.946 s | 6.142 ms |
| Female base | Original | 8.422 s | 4.634 s | 6.279 ms |
| Female base | Both relaxed | 9.196 s | 4.933 s | 6.154 ms |

All 16 prevalidated request runs performed zero FBX conversions. The live profile
HTTP round trip and source/hash checks are included in their time. Ordered selected,
rejected and unverified candidates and decision/source-binding fields matched live
serial screening in every body/hand setting. Offline preparation cost 8.44–10.01 s
per six-pose group and produced 24 certificates in total.

Independent Blender reimport compared 120 FBXs across 24 body/pose/hand cases:
mesh vertices, evaluated skin, topology, bind/posed bone matrices and weights
matched exactly within each case. Exporter metadata is excluded; raw FBX bytes
are not expected to match. The sanitizer preserves timings and fingerprints in
`experiments/hand_screen_prevalidation/2026-10-30/results.json`; private model,
BVH and FBX files are excluded from the repository.

After integrating the latest `develop`, the full inference suite passed 3,316
tests (10 skipped, 31 subtests); the
Converter contract set passed 395 tests (one skipped). Contracts cover mixed
certificate/live failures, presentation permissions, generated-byte isolation,
body/hand/runtime changes, pinned-file tampering and request cancellation.
The analysis deadline remains installed at every screening capacity (1–5),
and description rollout settings preserve the configured hand screen capacity.

These measurements are not the production 30-second session. Removing a 33.88 s
screening stage from the earlier 60.275 s example would leave about 26.4 s plus
profile/network overhead only if every required combination is covered and all
other work is unchanged. That is an estimate, not an end-to-end measurement.
AWS CPU/memory, fleet rollout, full-library coverage and custom shapes remain
unverified. No external model requests, AWS uploads or service changes were made.
