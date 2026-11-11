# Persistent final FBX preparation

`CONVERTER_FINAL_CACHE_ROOT` enables a private, persistent POSIX file store below
the existing `format=ready` preparation contract. It is disabled when unset. No
client change is required. This change neither uploads assets nor deploys services.

A hit returns the exact final FBX and its server-produced validation evidence,
without acquiring a Blender slot, retargeting, foot correction, shape baking,
rendering, or reimporting the file. The API still resolves registered characters
and approved shape sources, verifies expected source/model/preview hashes, and
checks the final binary digest. Every job receives its own BFF review key; those
keys, ownership, selection/hand revisions, current source availability, and late
response fences retain their existing checks. Only immutable bytes are shared.

## Inputs and invalidation

`library.json` contains allowed source BVH hashes and catalog character IDs/hashes.
The operator must set `CONVERTER_FINAL_LIBRARY_SHA256` to its exact SHA-256. A
missing, changed, unpinned or malformed registry cannot grant a cache hit. This
registry is generated from the active library DB/BVHs and authoritative catalog,
not request paths. Refined/private source hashes outside this registry use the
existing calculation and owner/job-scoped BFF artifact cache. Their pose is never
substituted by a library base pose.

The key includes source bytes, registered character identity and bytes, resolved
shape recipe/defaults/values and authoring source hashes, hand selections and
library hash, full/partial scope, mirror, baked camera matrix, selected hand
screen policy, Blender version/build, runtime policy switches and a code/policy
fingerprint. The fingerprint hashes the converter's Docker runtime allowlist,
Dockerfile and requirements, so batch and image must use identical sources. A
changed validator or solver invalidates previous entries even if its public
version label was not bumped. Source/model/authoring files are rehashed on hits.

The current declared defaults are `male-base-v0-8-0` (168.684899 cm, 7.943109
heads) and `female-base-v0-4-0` (169.893087 cm, 8.000001 heads); both have
head_scale=1. The batch reads every control default and approved shape source
from the supplied catalog and converter registry. It never assumes zero controls
or reuses a neutral/original FBX as a finished pose. `shape=null` still invokes the
registered approved anatomy recipe. Explicit default controls have a separate
key/shape receipt; no unproved equivalence normalization is applied.

Skin, outline, scene and interactive exporters do not call this cache. Rendering
and preview-model requests also bypass it. If any output-affecting combination
is not registered, exact, or eligible, the existing worker computes it. The
original source-finger transfer is applied before framing; explicit hand presets
still override it independently per side. A transferred finger stage now saves
and reimports the FBX and checks all transferred joint endpoints (100 µm limit),
in addition to its existing body-bone invariance check.

## Registration, resume and withdrawal

A per-key kernel file lock coalesces processes and releases automatically on
crash. Waits honor request cancellation/budgets. The successful normal runner and
final-reimport receipt are required before registration. Bytes are written and
fsynced to a same-filesystem temporary file, atomically renamed to a digest-named
file, and the metadata pointer is atomically published last. A crash before the
pointer leaves an unservable orphan; a retry rebuilds safely. `.state.json` records
building/failed/ready. Hash, size, binary format, exact binding and validation
metadata are checked on every read. Corruption triggers the normal path, never
serving the damaged file. Storage failure keeps the normal path available.

An optional `revoked.json` JSON array of cache keys or artifact digests disables
reuse immediately; update it by atomic replacement. This is cache invalidation,
not a replacement for library quarantine/deletion: the BFF's live source and owner
checks still determine whether a pose can be delivered at all. Job deletion
removes job receipts under the existing retention policy; private/refined inputs
are never registered here. Keep the directory private, with a storage quota and
an operator-managed lifecycle for obsolete policy namespaces. It is not an S3
backend and must not be mounted as a web static directory. Multiple replicas
share reuse only on the same supported persistent POSIX volume or separately
installed immutable offline assets. No distributed lock is claimed for S3/NFS.

## Batch procedure

Use the same converter image, registry and policy environment intended for serving:

```sh
python scripts/precompute_final_fbx.py --library /library --catalog /body/body-models.json --out /final-cache --camera '[[1,0,0],[0,1,0],[0,0,1]]' --plan
python scripts/precompute_final_fbx.py --library /library --catalog /body/body-models.json --out /final-cache --camera '[[1,0,0],[0,1,0],[0,0,1]]' --pose '2hand Idle_01'
# Resume the whole declared-default batch, serially (one bounded Blender worker):
python scripts/precompute_final_fbx.py --library /library --catalog /body/body-models.json --out /final-cache --camera '[[1,0,0],[0,1,0],[0,0,1]]'
```

These commands use the current semantic text search's exact front-reference camera.
Omit --camera for non-baked outputs; rough-search fitted cameras require their own
exact combinations and fall back when absent. Also run a separate
--explicit-defaults batch if a caller retains explicit default controls instead
of the BFF's current default-control normalization to shape=null. The two forms
are not assumed to have identical final bytes.

The plan lists all registered source IDs/hashes and the actual default bodies,
counts, catalog/DB/policy hashes, input registry pin and explicit time/size estimate.
It is written before expensive work. Use `--scope`, `--camera '<matrix JSON>'`, or
`--explicit-defaults` for separate exact combinations; unsupported/missing ones
fall back at runtime. Each plan also has a content-addressed filename. Resume
uses per-key validation, not a progress counter. `history.jsonl` retains previous
attempts; `progress.json` reports only this invocation and explicitly labels
unvisited combinations. Preserve both when reporting completed/failed/remaining
coverage. There is intentionally no upload or deployment option.

## Diagnostics and certification limits

Artifact-only responses add `final_cache` with hit/miss, reason, lookup/total time
and Blender conversion count. Existing structured conversion logs retain stage
timings. A hit has execution_ms=0. BFF exposes `X-Standin-Final-Artifact-Origin`
when present and persists this origin with its job-owned receipt; it describes
how those bytes were originally prepared, not whether a later BFF request hit
memory. Images are separately rendered from these exact final bytes.

The cache preserves the existing product safety gates. `final-fbx-v1` verifies
skin binding, finite geometry, skeleton and reimport integrity. Approved shape
recipes also report posed anatomy PSD and vertex/joint/weight roundtrip checks.
These do **not** constitute a complete whole-body penetration, containment or
folding certificate. The approved anatomy reports contact_qa.checked=false, and
the separate mesh validator's production topology/normal-policy registry remains
uncertified. Metadata explicitly records complete_collision_certificate=false.
This PR does not weaken or invent that missing certification. Refine research,
Top5 ordering and strict refine mesh certification are unchanged.

For a repeatable local contract proof, start `scripts/serve_final_fbx_reuse_probe.py`
with the batch's environment, `--library-sha256 <plan pin>`,
`--run-log <pilot>/http-converter-runs.jsonl` and `--forbid-conversion`. The server
binds only localhost and fails any conversion attempt. In the BFF checkout run
`npm run build`, then `node scripts/verify-final-fbx-reuse.mjs <pilot>`. Its stores
and confirmations are fixtures; the semantic presentation, BFF routes, Converter
HTTP, exact FBX/PNG and receipt fencing are real. Never deploy the probe.

Local measured evidence and remaining coverage are in
[the pilot report](qa/verified-final-fbx-20261111/README.md).
