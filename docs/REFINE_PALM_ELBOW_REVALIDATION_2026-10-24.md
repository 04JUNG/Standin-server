# Full-palm / rearward-elbow revalidation

The user's requested final result has **not** been achieved. The new candidate
moves the actual left elbow rearward by 31.6101 mm, passes the unchanged FBX mesh,
pose and replay gates, and does not worsen the fixed-camera 2D matching metrics.
However, its broad palm still does not contact the side waist. It is retained as
a **safe pose candidate**, with `full_request_passed: false`, not a contact success.
Gemini independently reports no clear visible improvement and palm contact not
achieved. No deployment, push, original-asset edit or replacement of prior viewers.

## Versions and actual inputs

- Branch: `codex/refine-palm-elbow-20261024`.
- Preserved baseline commit/branch: `814ea9730` / `codex/refine-depth-safe-backup-20261023`.
- Preserved previous thumb experiment: `64d18960b` / `codex/refine-surface-contact-20261023`.
- Baseline final FBX: `d0476ba2306700b5e66865682a05aefba0720781d28ccfbf7b935882ee810099`.
- Candidate final FBX: `01350422f07796422d8d40082d0ccba94c3e300cf260ee7194f85129133e6fb4`.
- Actual rough: `124629.png`; the newly attached rough has identical RGB pixels
  to the image used for the saved observations (288 × 630). File hashes differ.
- The newly attached side-view image was visually inspected as context, not used
  as a replacement geometric reference or ground-truth depth observation.
- Same real BVH `rokoko_FootTapping_mixamo_00040`, source SHA
  `835b3eb6ab103385ce8b9ee1627014df950ec36671e20108f80e63008f3c0d17`.
- Female BASE `female-base-v0-3-4`, left `support`, right `relaxed`.

## Changes

`palm_waist_contact.py` identifies the anatomical palmar side from the wrist and
finger roots, excludes digit groups and back-of-hand faces, and freezes a broad
palm in four quadrants. The target is the same-side waist surface, not the entire
abdomen or the nearest arbitrary body part. The actual selected palm is about
5,414 mm²; a small thumb patch cannot qualify. Contact requires 0.5–5 mm positive
sample clearance, opposed normals (cosine at least 0.7), at least 80% overall
qualified face area, at least 60% in each quadrant, and the separate full-mesh
certificate. Face vertices and centroids must all qualify. This is a broad mesh
sampling contract, not a continuous flesh pressure/compression simulation.

`plan_palm_elbow_contact.py` jointly uses the actual elbow/wrist 2D observations,
a rearward elbow constraint in the rig's anatomical coordinates, broad palm
surface gap and direction. SLSQP runs from three deterministic seeds. Body,
selected-arm, elbow and wrist errors must not worsen; the plan reserves a small
1e-5 NME margin for final serialization. All original angular, twist, depth and
bone-length guards still apply. Protected wrist rotation remains protected.
A second elbow-prioritized objective converged to essentially the same boundary.
No changed source pose was silently used to reset the same candidate's limits.

`verify_palm_elbow_candidate.py` runs actual Blender realization, independent
skin planning and full-body QA. `audit_palm_elbow.py` then verifies both byte-bound
receipts, regenerates the frozen palm regions from the real baseline FBX and
measures the actual final FBX. A safe pose alone cannot set `full_request_passed`.

## Actual result

Source run: `/private/tmp/refine-palm-elbow-actual-v1-20261024`.
Planning run: `/private/tmp/refine-palm-elbow-plan-v1-20261024`.
Local comparison: `http://127.0.0.1:18773/palm-elbow-20261024/`.
Durable run: `/Users/dowon/.codex/visualizations/2026/10/22/01a162e8-7b84-7af3-995a-87b9efbe7b26/refine-palm-elbow-run-20261024`.

| Fixed camera NME (lower is better) | Baseline | Actual candidate |
|---|---:|---:|
| Body joints | 0.1072304744 | 0.1072207235 |
| Selected arm | 0.0911489896 | 0.0910902037 |
| Left elbow | 0.0289648163 | 0.0289555537 |
| Left wrist | 0.1533331630 | 0.1532248537 |

Overall reduction is only about **0.0091%**. It satisfies non-regression, but is
not evidence of a meaningful perceptual pose-matching gain or a 3D accuracy rate.
The same camera, projected torso scale and hip center are used for both files.

Actual anatomical rearward displacement is **31.6101 mm**. The elbow remains
about **36.51 mm in front of the anatomical shoulder plane**, versus 68.12 mm
before. This is not Blender world Z (height) or a claimed observed image depth.
The original source elbow would need at least about **13.67°** of unrestricted
upper-arm rotation just to reach its anatomical shoulder plane; the unchanged
upper-arm correction cap is 12°. That geometric bound concerns this starting
rig and plane only. It is not a proof that all possible source poses are infeasible.

Final arm corrections: upper arm 6.5729° with 3.8286° axial twist; forearm
11.9900° with 0.1140° axial twist. Maximum camera-depth drift is 0.0540 torso
lengths, below the existing 0.15 bound. No angular/depth policies were relaxed.

Neutral/body/hands full-mesh checks passed in 2.23 / 2.38 / 2.37 seconds (3-second
limit). Independent vertex replay errors are 0.44 / 1.73 / 2.38 µm (10 µm limit).
Actual body/final rig matrix errors are 0 / 0 (1e-5 limit). No crossings, folded
faces, contained components or unknown safety reasons were accepted.

The side-waist full-palm audit **failed**:

- Qualifying palm area: **0%**.
- Minimum sampled gap: **47.55 mm**.
- Median centroid gap: **104.04 mm**, reduced from about 141.03 mm in planning.
- Four quadrant median gaps: **75.43 / 84.57 / 122.87 / 122.71 mm**.
- Status: `palm_contact_blocked`, reason `broad_palm_contact_missing`.

## Other genuine library candidates checked

The already retrieved real candidates were converted without changing the frozen
converter or authoring a new source to bypass a trust bound. Both checks were
rejected; these files were not delivered as safe outputs:

| Candidate | Actual body NME, same baseline camera | Safety outcome |
|---|---:|---|
| `b2_hand_on_hip_talk` | 0.14505441 | `discrete_shell_orientation_reversed` |
| `rokoko_PlayingSaxophone_mixamo_00539_mirror` | 0.12702013 | `discrete_shell_orientation_reversed` |

These are diagnostics of the tested candidates, not an exhaustive search or proof
that a suitable different source can never be found. Their raw body outputs were
not repaired/adopted after already failing the non-regression check.

## Live Gemini

The first `gemini-3.8-flash` request timed out after a 45-second read limit. A
second explicit request with a 90-second limit returned a response. The attachment
and six before/after actual FBX renders were sent; all image hashes are bound in
the review. Credentials were read in memory and not logged.

- Preferred: `uncertain`.
- Elbow rearward / elbow rough match: `same` / `same`.
- Broad palm side-waist contact: `not_achieved`.
- Overall pose change: `same`.
- Confidence: `high`.

It notes that the elbow is not sufficiently behind the torso and the hand remains
in front of the lower abdomen. This agrees with the unmet visual objective despite
a measurable small rearward displacement. VLM evaluation never issues safety
or 3D ground-truth authority.

## Verification and scope

- Converter suite: **506 passed, 1 skipped**, with the same three Blender-only
  module exclusions as previous runs; actual Blender work was executed separately.
- New tests reject thumb-sized patches, partial/quadrant-only contact, back-of-hand
  normals, penetration, threshold overrides, duplicate faces, and false rearward
  claims; backward movement cannot excuse a worse wrist/pose match.
- Frozen converter: all 13 file hashes unchanged. Python compile and diff checks.
- Separate before/after viewer, actual joint depth plot, contact audit, Gemini
  report and full trace. Previous baseline and thumb pages remain separate.

The requested complete natural palm/side-waist result is still unresolved under
these tested inputs and unchanged limits. Do not label this candidate as a full
contact success, promote it automatically, or claim the tiny NME change establishes
visible improvement. More appropriate source poses and a contact-aware mapping
need evidence-backed evaluation; expanding the current limits is not an accepted
substitute.
