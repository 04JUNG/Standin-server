# Cached limb diagnosis — experimental, disabled by default

This PR does not hook diagnosis into `/analyze`, `/refine`, the parts worker or
the app. The native experimental runner exits before importing Blender or
opening assets unless `--enable-experimental-diagnosis` is present. Existing
production defaults and API/schema contracts are unchanged. No external VLM
SDK or model is called. `release_allowed` remains false.

The pilot reuses the original cached VLM description/hands for the same hashed
image, reads the selected native FBX bones/skin, diagnoses supported differences,
selects existing arm recipes, and verifies the actual saved result. A missing
side review does not erase an explicitly named anatomical limb. Naming a limb
does not fabricate a visible shoulder–elbow–wrist/hip–knee–ankle chain.

## Observation and execution boundaries

- `cached_targets` / `explicit_limb_text`: bounded Korean text adapter. Isolates
  left/right clauses, supports explicit bilateral arm observations, retains
  evidence source spans, distinguishes waist placement from an overhead raise,
  head side/back and unsupported object holding. It is not general natural
  language inference. Unknown direction stays unknown; explicit abstention,
  contradictions and screen-relative sides hold.
- `diagnosis` / `native_state`: compare actual native directions, hand folding
  and contact patches. `match` means only supported measured constraints match.
  Chin support holds until whole-hand placement/orientation can be checked;
  contact by a small thumb patch alone previously yielded a false match.
  Paired arm goals hold until an atomic pair comparator/recipe is connected.
- `experimental/cached_leg_diagnosis` / `native_leg_diagnosis`: preserve named
  and bilateral leg goals independently of arm side reviews. Compare shape,
  crossing and relative knee heights; no ground/table plane is inferred from
  the lowest vertex. Support geometry missing means contact held. The unified
  run performs read-only leg diagnosis; it never automatically applies legs.
- `cached_limb_inventory`: multi-person cache order is not identity. Uses
  unique source-box IoU>=0.8 and margin>=0.15, rejects competing assignments,
  and never substitutes another person's P1. Single-person identity requires
  one expected person and one cached person in the same hashed image. These
  correspondence tolerances are conservative development settings.

Selected arm goals must match again after export. An incompatible canonical
surface or a failed fit rejects the candidate and retains the exact source.
The independent export verifier checks contact, directions, bone lengths,
protected joints/surface and new/worse skin. Final measurements are refreshed
from its actual admitted/retained FBX, avoiding stale rejected-candidate data.
Native contact prerequisites include the straight forehead recipe, exact
topology canonical surface rebuilding and exact pose-channel rollback. Existing
collision tolerances are not loosened. Other added hand/arm recipes are native
opt-in support; diagnosis does not invent unsupported goals to use them.

## Reproduce locally

Workspace assets and Blender must already exist; images/FBX/model data are not
redistributed in Git. The frozen v3.2 manifest is used only for image/person/
selection/asset lineage. No reviews, rubric grades or correction answers are
passed to target extraction, selection or fitting. The output folder must be new.

```text
python scripts/prepare_cached_limb_eval.py <workspace> <new-run>
blender -b --factory-startup --python-exit-code 1 --python scripts/run_cached_limb_pilot_native.py -- <repo> <workspace> <new-run> --enable-experimental-diagnosis
blender -b --factory-startup --python-exit-code 1 --python scripts/verify_arm_language_batch.py -- <repo> <workspace> <new-run>
blender -b --factory-startup --python-exit-code 1 --python scripts/verify_cached_limb_eval.py -- <repo> <workspace> <new-run>
blender -b --factory-startup --python-exit-code 1 --python scripts/render_cached_limb_pilot.py -- <repo> <workspace> <new-run>
```

Review applied native renders and record `visual-audit.json` separately from
diagnostic inputs. Mark unchanged cases as byte-preserved, not visually correct.
Publish with `python scripts/report_cached_limb_eval.py <new-run>
<new-run-parent/index.html>`. Native three-view images and input/target/plan/
saved-verdict links remain local. The legacy six-person preparation/report
scripts are also included; native execution still requires the explicit flag.

Tests: `python -m pytest tests/test_arm_diagnosis.py
tests/test_explicit_limb_text.py tests/test_cached_limb_inventory.py
tests/test_cached_leg_diagnosis.py`. These cover negative/ambiguous evidence,
no cross-arm action leakage, default-disabled execution without Blender/assets,
missing/permuted people, paired abstention and already-correct constraints.
`check_arm_quality_native.py` supplies native failure-injection, direction and
surface regressions on the existing authored quality fixtures, independently of
the cached-target pilot. That manual fixture is not recognition evidence.

## Expanded frozen-set evidence

Local gallery: `outputs/standin-auto-diagnosis-expanded-20261030/index.html` in
the parent workspace. Compact evidence is in
`experiments/cached_limb_diagnosis/2026-10-30/`.

- Inventory:50 cuts/60 expected people;52 cached observations correspond and
  can be structured (this does not mean52 confident/correct goals).28 people
  have both reliable correspondence and a verified frozen native FBX.30 lack
  a frozen selected FBX record and8 have ambiguous/missing cache correspondence;
  these reasons overlap, leaving32 without native execution.
-28 native people:7 partial arm applications,21 exact byte-preserved outputs.
  The7 are T002/T015/T018/T024/T027/T029/T034. Waist placement added3 applications
  beyond the six-cut pilot. Every selected supported goal matches after saved
  reimport.15 arms whose supported conditions already matched were not selected;
  repeating diagnosis on the7 applied files requests0 further corrections.
- Actual applied-case native time:0.68–3.02s; basic four-limb read median0.31ms.
  Times exclude fresh recognition, Blender startup, previews and queue/network.
- Saved outputs have no new/worse skin warnings. Existing unrelated warnings
  and within-hand collision limitations remain. Unselected matrices retain the
  existing2e-5 tolerance, not a loosened threshold.
- Same-agent render audit confirms only partial goal/contact improvement for
  the7 applied people. Opposite phone/fist hands, coupled poses, obscured
  observations, head/eye alignment and full-pose accuracy remain unresolved.
  T026 demonstrates the difference between correcting the adapter and correcting
  a wrong original VLM observation. Independent accuracy, score gain and a
  zero false-positive rate are **not measured**.

Do not turn these engineering checks into a90% accuracy claim. Next promotion
requires independent reference-vs-result assessment (including already-correct
poses) and, after an explicit app connection, end-to-end file/preview/save
identity checks on develop/staging. This PR does not deploy or merge itself.
