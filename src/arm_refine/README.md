# 팔 포즈 파츠 · 로컬 MVP

The opt-in cached limb diagnosis pilot and full-set replay protocol are documented
in [CACHED_LIMB_DIAGNOSIS.md](../../docs/CACHED_LIMB_DIAGNOSIS.md). It is disabled
by default and does not change the app/worker execution paths.

04J-318. 기존 BVH에서 쇄골 앵커 → 어깨 → 팔꿈치 → 손목 체인을 추출한다.
독립 worktree `feat/04j-318-arm-refine`, 기준 커밋 `2eaae51`.
공통 API/스키마/검색/IK/클라이언트는 수정하지 않는다. 운영 경로에 연결하지 않았다.

## 산출물과 사용

`python scripts/build_arm_parts.py <기존 BVH 폴더> <새 출력 폴더>`

Python + numpy + scipy + Pillow가 필요하다. 출력 폴더가 이미 있으면 덮어쓰지 않는다.
12개 수동 선정 포즈에서 왼팔12/오른팔12/양팔12, 총36개를 만든다.
`catalog.json`, `parts/*.json`, `parts/*.bvh`, 정면/측면 스켈레톤 PNG, `index.html`을 쓴다.
새 파츠를 추가하려면 STARTER에서 출처 BVH ID를 명시한다. 원본·DB에 쓰지 않는다.

JSON은 적용할 체인/출처 해시/채널 순서/BVH bind offset·행렬/현재 local·world 행렬,
몸통 기준 방향·길이·앵커·손목 회전을 담는다. BVH는 팔만 있는 실제 독립 스켈레톤이다.
그 BVH의 `ArmPartAnchor`는 프리뷰·관절 교환용 합성 root이며, 전신 BVH를 대체하면 안 된다.
손가락·손 메시를 생성하지 않는다. BVH zero-channel bind는 원본 BVH 기준이며 FBX의 bind를
추정한 값이 아니다. 어댑터는 실제 목적 FBX의 rest/pose와 뼈 길이를 사용한다.

```python
from src.arm_refine.parts import read_part, apply_part, search_parts
part = read_part('parts/reach_up--right.json')
new_frame, report = apply_part(base_joints, base_frame, part)
# Blender에서는 src.arm_refine.blender_adapter.apply_part(arm, meshes, part)
# 검색은 {left/right: [상완 방향, 전완 방향]} 몸통 기준 수동 query를 받는다.
```

좌우는 자기 기준이다. 메시를 잘라 붙이지 않고 관절 방향을 수송한다.
목적 쇄골/어깨 위치·뼈 길이·비선택 관절을 고정하고, 선택된 상완·전완 회전과
손목 보상 회전만 변경한다. 손목 world 회전과 손가락 local pose를 유지한다.
양팔 관계가 있는 paired 파츠는 한 팔만 적용할 수 없으며 한 실패에도 전체를 되돌린다.
목적 체형에서 팔짱의 실제 접촉까지 보장하는 제약은 아직 없다.

Blender 어댑터는 `scene_refinement.rig.validate_rig`, 기존 converter 출력/재임포트,
선택적 화면 목표에 `scene_refinement.arms.solve`를 사용한다. 기본값은 이미지 목표 없이
파트 방향 수송이다. `connection_deg`는 몸통 축 조정(각 축 ±45° 한도)이다.
이미지 목표·연결 각도 옵션은 기본 0° 파츠 적용 경로와 별도로 추가 검증이 필요하다.

## 검증과 한계

`python -m pytest tests/test_arm_parts.py -q` : 17개.
좌우/양팔, 여러 Euler 순서·단위·부모 회전, JSON/BVH 재읽기, 손목·손가락·다른 부위 보존,
양팔 원자성, 충돌 게이트 폐기, 검색 입력 불변을 검사한다.

Blender: `--background --factory-startup --python-exit-code 1 --python scripts/validate_arm_parts_fbx.py -- <repo> <v3.2 package> <parts folder> <새 검증 폴더>`
CB049 한 팔 좌/우, CB001 양팔은 실제 FBX로 undo·blend 저장/로드·FBX export/reimport를 검사했다.
CB027(T042)은 동결 FBX가 없어 not_run이다. 다른 자산으로 바꿔 동결 전후 비교라고 주장하지 않는다.

이것은 수동 선택한 팔 파츠와 교환 경로의 공학 검증이다. 자동 부위 탐지, 24개 컷 전수 개선,
재채점 또는 성공률 측정이 아니다. 카메라/가림의 확실한 근거 없이 3D 오류를 단정하지 않는다.
`search_parts`는 수동 3D 방향 질의이며 검출기나 텍스트 검색이 아니다.
충돌은 기존 관절 capsule proxy이다. 어깨 twist 관절 한계·피부 접촉은 미인증이다.
잘린 상반신 FBX는 팔을 옮길 때 기존 절단면이 드러날 수 있어 실제 메시 검수가 필요하다.
후속 완성 경로는 전체 원본 리그에 적용 → 기존 body scope로 재절단 → 동일 자산 검토이다.

다음 설계: 사용자 부위 선택 UI, 가림/자기 좌우 보류를 포함한 관측 증거 어댑터,
원본 전체 메시에서 적용/재절단, 양팔 접촉·상체 관통 제약, BFF reviewed artifact 저장 연결.
기존 평가 정답과 점수 시나리오는 입력에 사용하지 않았다. 결과는 `release_allowed=False`다.

## 라이브러리 품질의 썸네일

기본 파츠 목록은 실제 메시 512×512 썸네일을 사용한다. 정면/사선/측면/후면을 제공하고,
선택한 팔만 보는 모드와 원본 인형에서 보는 모드를 전환한다. 스켈레톤은 보조 보기이다.
스타일은 기존 `approved-anatomy-detail-dark-5scope-8view-v1`과 같은 studio 재질·cavity·
조명·배경·AA16을 사용한다. 팔만 보는 이미지는 해시가 검증된 원본 FBX에서 선택 팔의
skin weight 경계를 잘라 렌더한다. 이 절단은 미리보기에만 사용하고 파츠 JSON/BVH는 바꾸지 않는다.
원본 손가락은 시각 참고용이며 파츠 payload 또는 손 모양 교체 기능으로 해석하지 않는다.

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/render_arm_part_thumbnails.py -- \
  <repo> <parts-package> <library-render-root> <new-thumbnail-output>
python scripts/publish_arm_thumbnails.py <parts-package> <new-thumbnail-output>
```

`build_arm_parts.py --native-thumbnails <verified-thumbnail-output>`으로도 연결할 수 있다.
다른 포즈·좌우·스타일·해시·뷰가 섞이면 attach를 거부한다. 인형 참고 이미지는 library
scope8 report/metadata/image 해시까지 확인하고 복사한다. 캐시 미스 때 다른 포즈 이미지를
대신 쓰지 않는다. 해당 source FBX와 검증된 라이브러리 이미지가 필요하다.
이번 실행은 팔 이미지144장 + 원본 인형 이미지144장, 총288장 모두 해시·512px·비공백 검증을
통과했다. 출처 혼합 방지 테스트7개를 더해 관련 테스트는24개이다.

## 관계 태그·팔짱 품질·필요 파츠 연구

`semantics.py`는 기존 normalize_text를 재사용해 부위 범위의 동작/자기 좌우/손목 높이/
팔꿈치 굽힘/양팔 관계/접촉 대상과 근거를 저장한다. Python과 `text_search.js`를 같은
10개 질의로 비교했다. 현재는 제한된 별칭·규칙 검색 시제품이며 임의 자연어의 이해도나
임베딩/LLM 태깅 성능을 측정한 것이 아니다. `tag_arm_parts.py`가 출처 해시를 확인한
기존 시각 설명과 BVH 관절 관측을 결합한다. 설명에 없는 접촉을 파일명만으로 확정하지 않는다.

팔짱·방어·손 편 팔 포개기를 구분한다. 팔짱은 양팔 묶음만 검색하고, 손을 안쪽으로 숨기는
유형은 미검증이므로 반환하지 않는다. 보류 상태의 fold_arms 파츠는 두 apply 어댑터 모두
변경 전에 거부한다. 다른 인체로 수송한 후의 품질은 별도 측정이 필요하다.

`pair_skin.py`는 실제 evaluated skin의 손바닥 샘플과 반대쪽 위팔 삼각형 사이 거리를 측정한다.
12mm/샘플비율은 로컬 후보 선별용 실험 기준이며 사람 자연스러움 점수나 체형 공통 인증값이 아니다.
기존 `pose_curation.rendering.anatomy`와 `self_collision.py`의 기준을 그대로 재사용했다.
`measure_arm_pair_quality.py`, `fit_folded_arm_candidates.py`, `evaluate_fold_candidates.py`는
출처 파일/기본 master/기존 충돌 모듈을 읽고 별도 산출 경로에만 기록한다.
후자는 명시 옵션의 선택 쇄골 전진을 포함할 수 있으며 몸통·머리·다리·손가락을 유지한다.
이번 보정27개 중4개는 기존 피부·손목·해부학 검사를 통과했으나 실제 반대 위팔 접촉은 부족해서
모두 팔짱 완성 후보로는 보류했다. UAL2 원본도 피부 경고가 남아 엄격 추천에 올리지 않았다.

`research_arm_needs.py`는 사용자가 지정한 `정확도(90%) 5sp` 세션과 동결 v3.2 근거를
연결해8개 주된 오류 유형과24컷(한팔14/양팔10)의 필요 파츠 목록을 만든다.
이 판정 목록은 평가·계획용이며 detector 입력으로 사용하지 않는다. 사진은 원문 링크로만
연결했으며 라이브러리나 학습 자료로 편입하지 않았다. v2는37개 파츠의 태그·품질 보류 UI와
검색 시제품이다. v1 및 동결 평가/원본 라이브러리는 보존했다.
# Arm expansion v3 (2026-10-28)

`scripts/expand_arm_parts.py` versions v2 into 60 parts: 23 additions from visually inspected existing library sources, including actual active-side mirrored poses. Added families cover hands on hips, chin/face with a low elbow and opposite elbow support, back of head/neck, extended forward/up reach, asymmetric front/back arms, object-holding context and low arms behind the back. This is a curated candidate set, not measured coverage of all 24 evaluation failures. In particular, two palms supporting the chin remain absent.

Text intent routing now separates these families before lexical ranking. “양손 허리” rejects a pose with only one hand on the hip; “양손으로 턱” does not substitute a hand on the face plus an elbow-supporting arm. Explicit full extension requires an elbow flexion at most 30 degrees. Python and browser search agree on the 13 package queries. The 60 JSON/BVH assets roundtrip within 1e-7 torso units, and 480 512px native/source thumbnail hashes are verified. New nonfold additions have not received full skin collision screening or human approval; source image inspection is not contact certification.

`scripts/fit_folded_arm_candidates.py` now supports independent wrist levels, rising hand directions, optional actual-surface fitting and an explicit preview-only finger curl preset. Sixty additional authoring probes were preserved in `fold-surface-fit-v4` through `v9`. None passed both actual skin collision and palm-to-upperarm gates. The visual prototype selected for comparison is v9/case 05: it reduces the flat hand silhouette, but retains left forearm/right hand 14.9mm and left upperarm/right hand 22.8mm collision warnings. It is not a released arm part and cannot satisfy strict folded-arm retrieval. Its edited fingers are not transferred by `apply_part`.

`scripts/render_fold_comparison.py` renders the source, rising-hand intermediate and curled-hand prototype in four views under `natural-fold-review-v1/`, with their original receipts and editable blends. `scripts/validate_arm_package.py` validates geometry, sources, native images, static links and search parity. UI interaction remains not_run due to the file-URL browser policy. The original library, converter, evaluation, IK and production schemas remain unchanged.

## Photo-guided asymmetric fold revision

The supplied photo clarified two different hand roles: the front arm's hand goes below/behind the opposite upperarm, and the lower arm's fingers rise along the other upperarm. `scripts/author_tucked_fold.py` authors independent wrists, elbow poles, palm directions and explicit finger poses. `src/arm_refine/fold_fitting.py` offers a sampled skin fitting heuristic; optional full containment/collision and front-occlusion terms refine it. The unchanged original anatomy, wrist and full weighted-skin checks remain the final screen. Landmark occlusion does not certify that an entire hand is hidden or correctly touching.

The selected source is `tucked-layers-v12/tucked_pronated_fit_00`: original linear skinning, zero anatomy/skin warnings, wrist swing/twist guards passed, protected local matrices unchanged. Its forearm rolls are 66 and -40 degrees; larger-roll and volume-preservation experiments were excluded. It is a visual-review candidate, not a naturalness score or human-approved tucked-hand asset. Some fingertips remain visible. A subsequent palm-focused candidate retained a collision and was excluded. The photographed relation involves a tucked hand and rising fingers; do not interpret the older two-palms predicate as a certificate for that relation.

`scripts/export_tucked_fold.py` exports the selected whole pose to `photo-fold-source-v3/`. Removing micrometre residual local translations/scales changes joints by at most 1.11e-5 m, restores rotation-only BVH channels, and passes the unchanged independent FK check (52 joints, 3.25e-5 cm maximum error). FBX reimport changes joint positions by at most 4.1e-7 m and again has no anatomy/skin flags. Whole-hand/finger skin distances are recorded separately from palm-only distances; nearest distance alone is not contact certification.

`scripts/publish_tucked_fold.py` packages the original user photo, previous and revised four-view renders, original receipts, editable poses and verified exports in `photo-fold-review-v2/`. `arm-pair.candidate.json` is marked as a folded-arm transport hold: the authored clavicle/wrist/finger pose is necessary, while the default arm-only apply preserves those destination poses. Automatic application and strict tucked-hand retrieval remain blocked. The 60-part library and all production schemas/converter defaults remain unchanged. Artifact links/hashes and JavaScript syntax were checked; browser UI remains not_run under the file-URL policy. The existing 36 arm tests pass.


## Interlocked-hand correction remains unfinished

The user rejected the v12 source despite its zero collision flags: the rising hand was outside/below the upperarm, rather than emerging between torso and upperarm. A clean mechanical screen did not establish the intended gesture. The older `photo-fold-review-v2` must not be treated as visual approval.

The author now supports real hand-library presets with gathered fingers, bounded preset flexion, clavicle depression, rising-hand constraints and a sampled opposing-surface insertion heuristic. The latter is diagnostic only: hidden landmarks and opposing nearest surfaces do not establish an exterior, collision-free insertion. Full original skin/anatomy checks and actual four-view inspection remain necessary.

`photo-fold-review-v3` preserves a work-in-progress interlocked candidate from `interlocked-fold-v23/interlocked_relaxed_shoulder_02`, with original reports and editable/exported assets. Its hand/arm penetrations remain 28.8, 10.0 and 37.7 mm. Both wrist guards pass, but the skin screen fails. It is explicitly unfinished, is not a replacement library part and is not eligible for automatic application. No candidate in this interlocked study is accepted as complete. Original master, library, evaluation and converter defaults remain unchanged. The 36 existing arm tests pass.

## Measured blockout and frozen-case arm experiment (2026-10-29)

The new fold authoring path separates volume feasibility, hand routing and appearance.
`export_fold_blockout_inputs.py` measures the unchanged native master's bone lengths,
torso hull and arm cross-sections and saves actual gathered-finger hand forms.
`volume_blockout.py` builds articulated oval arm proxies and hand rails from those
measurements. These are approximate authoring volumes, not exact deformed skin.
The torso convex hull, 95th-percentile sections and rail sampling can differ from
the native surface. Proxy feasibility must still be followed by native authoring,
the original full skin/anatomy/wrist checks and multi-view visual review.

`solve_fold_blockout.py` first minimizes constraint slack to assemble the volumes;
that provisional repair is not an admissible pose. Appearance optimization runs
only after safety admission, with separation and joint bounds as constraints.
The hand-role gate separately requires the hidden fingertips behind the opposing
upperarm and the other hand emerging upward through the torso/upperarm gap.
Only an admitted block may write `native-case.json`. Solver success alone cannot
override failed clearance, hand routing or bone lengths. `render_fold_blockout.py`
shows coloured proxies explicitly as blocks, never as a finished character.

The latest `outputs/standin-arm-refine-20261029/volume-first/oval-contact-block`
failed, with a maximum proxy-clearance violation about 12.3mm. That number is not
a measured native-skin penetration or proof that the photographed pose is
impossible. No final folded-arm part was emitted or applied. An earlier
`contact-assembly` proxy passed an older mathematical route but looked wrong;
it is not visual approval. The interlocked fold remains unfinished.

The user's scoring request is a separate, explicitly **manual** experiment on
three known arm failures from the frozen v3.2 run. `arm_score_render.py` copies
the exact original FBX/input/review, applies one existing arm part and renders it.
`contact_arm_probe.py` then places the actual hand skin against a fixed torso or
head surface and solves only that arm with fixed bone lengths. A small clearance
sequence is screened by the original full skin checker. The chin example also
uses the existing real hand-library fist preset; other selected finger poses are
preserved. Nonselected body, head, legs, clavicles and opposite arm stay fixed.

`verify_arm_score_exports.py` reimports the actual exported FBX and independently
measures target error, segment lengths, wrist axis, protected matrices and skin.
All three final exports have zero new/worse skin flags, under 0.5 micrometres of
wrist goal error and protected matrix differences below 2e-5. Existing opposite
arm/body flags in T015 and crouch-leg flags in T030 remain. T026 has no skin flags.
The checker does not cover within-hand finger-to-finger collisions.

`render_arm_score_pairs.py` uses union bounds of all three actual FBX variants to
render each case at the same scale/camera in 0/45/90-degree full and arm views.
`report_arm_score_experiment.py` copies the frozen rubric/schema/scorer byte for
byte into the separate report, validates the review schema and calls the original
`score_case`. Only the selected arm criterion changes; all other judgments and
upper-body inactive keys remain exactly unchanged.

| Case | Scope | Original | Part only | Surface correction |
| --- | --- | --- | --- | --- |
| T015 / CB010, left hand at waist | full / 100 | 85 | 85 | 95 |
| T030 / CB049, left hand at head | full / 100 | 90 | 90 | 90–100, uncertain palm contact |
| T026 / CB036, right fist under chin | upper / 45 | 30 | 30 | 40 |

These are AI partial reassessments, not human-reviewed scores, a full 50-case
rerun, production search evaluation or evidence of overall 90% accuracy. The
45/100 scopes are not averaged together. The score interval is a logical bound,
not a probability or confirmed 100. T020 / CB022 crossed arms is held at its
original 75–80/100 and is not an executed fourth replacement.

Final artifacts are in `outputs/standin-arm-refine-20261029/arm-score-run04/`:
`index.html`, three review JSONL files, the unchanged scoring contract, images,
actual FBXs, editable blends, camera hashes and independent receipts. The
packager checks all local links and 54 nonblank 640px native renders. Browser
interaction remains not_run under the file-URL policy. No original evaluation,
library, production converter, shared IK or schemas were changed.

Historical runs remain intact. Run02's T030 `accepted` field was wrong: when no
eligible candidate remained, an old selection path kept the last collision-free
attempt despite a 99.38-degree wrist. `retrospective-audit.json` records its
rejection. The corrected admission requires explicit skin, wrist, length and
target evidence and never promotes the last failed attempt. Four regression
tests cover this and the failed block gates; the 40 relevant arm tests and the
54 existing smoke tests pass.

## Common contact-part application (2026-10-29)

The user rejected the angle-only transfer's raised/floating hand. The replacement
experiment now uses a separate `standin.arm-contact-part.v1` procedural catalog
in `contact_parts.py`: waist thumb-pad contact, head-side central-palm contact,
and a central fist support patch under the chin. These are newly authored
anatomical recipes, not claims that the older 60 BVH parts already carried those
constraints. Head side is not silently substituted with back-of-neck/high-elbow
intent. The small explicit alias selector is not an image intent detector.

`contact_adapter.apply_contact_part` takes a recipe, anatomical side, destination
rig/mesh, canonical rig profile and native skin checker. The engine has no case
ID, reference image, score/review input or world-coordinate override. It finds
the target on the destination surface, sets the selected hand shape, and fits a
hand contact frame plus fixed-length elbow swivel. The fit minimizes geometric
constraint violation before appearance, with coupled tilt/clearance steps to
move along a contact boundary. It does not render a grid of poses to pick one.
Admission separately requires actual central-patch median/p90 distances,
contact-normal alignment, elbow placement, wrist axis/swing/twist, bone lengths,
protected matrices and zero new/worse native skin flags. Missing evidence fails.

Head-side contact uses an open hand with the thumb's actual bones projected into
the palm plane. A thumb curled toward the contact surface had caused a 16mm head
penetration even while the central palm was close. Waist contact instead uses
the thumb pad with a relaxed hand; it is not certified as full-palm contact.
The chin template uses the existing fist preset's central top support area,
rather than requiring the whole wide fist rim to contact a narrow chin.

This path deliberately changes the selected wrist and fingers. The clavicles,
body, head, legs and opposite arm remain fixed. It is separate from the existing
v1 angle-only adapter and does not change production refine's wrist/finger policy.
The canonical wrist profile is read from the unchanged Standin master: imported
frozen FBXs already have their pose baked into the bind skeleton, which must not
be mistaken for anatomical neutral. Other rigs/body shapes are not certified.

`run_contact_parts.py` consumes only explicit intent/side and immutable source
paths/hashes; the same function runs all requests. Failed candidates are rolled
back, including fingers, and stored separately as rejected diagnostics. Baking
changes the bind pose, so diagnostics and final results are exported from their
own unbaked scenes. Historical run01–04 diagnostics had early implementation
defects and are not promoted as results. Final evidence is **run07**, under
`outputs/standin-arm-refine-20261029/contact-parts-v1/`.

`verify_contact_run.py` reimports actual FBXs and measures contact, protected
matrices, lengths and the original full skin checker again. `render_contact_run.py`
renders the original, legacy angle-only and common-contact exports with one
camera per view, at 0/45/90/180 degrees in full and arm crops. All 72 native
640px images are validated. `check_contact_invariance.py` verifies the same
head-side recipe after global rotation/translation, rejection of an unreachable
head target, and full rollback after a deliberately injected skin-checker error.
These are geometry/transaction tests, not unseen-pose accuracy measurements.

| Case | Original / legacy | Common contact, AI arm-only reassessment | Contact patch median / p90 |
| --- | --- | --- | --- |
| T015 / CB010 | 85 / 100 | 95 / 100 | thumb pad 7.1 / 14.0 mm |
| T030 / CB049 | 90 / 100 | 100 / 100 | central palm 4.4 / 6.5 mm |
| T026 / CB036 | 30 / 45 | 40 / 45 | central fist support 5.6 / 10.5 mm |

T030's previous manually corrected palm median/p90 was 35.7/41.6mm; the new
result removes the floating-palm reason for its earlier uncertain judgment.
The v3.2 score updates only the selected arm; 100 here carries forward all other
original judgments and is not a fresh full-body quality certificate. These three
cases were used during development; no held-out or full 50-case accuracy is
claimed. Head mismatches and upper-body inactive keys stay unchanged. No average
combines the 45/100 scopes. Existing other-body collision flags remain; within-
hand finger self-collision is outside the original checker.

`report_contact_parts.py` alone reads the separately saved AI judgments and
frozen reviews, copies the exact rubric/schema/scorer, and creates `index.html`.
The solver never reads that data. Nine review records validate, nonselected
judgments remain byte-for-byte equivalent, source/camera/image hashes and local
links pass, and report JavaScript parses. Browser interaction remains not_run
under the file-URL policy. Four more regression tests cover contact-family
separation, paired-fold rejection, floating-palm admission and bad contact/arm
conditions (44 relevant arm tests total). No production or original library
files were changed. A validated atomic two-arm contact recipe is still absent;
fold requests fail before independent one-arm application and remain on hold.

## Expanded contact trial: 13 native cuts

The next user request expanded the experiment beyond three cuts. The full
24-cut arm-failure inventory has 14 verified original FBXs and 10 image-only
records. T020's interlocked fold and T022's noncontact asymmetric arms do not
have a compatible validated contact recipe. The other 12 native error cases
were scheduled, plus T027 (an already-matching forehead arm) as a regression
control. Cohorts are three repeated development cases, nine additional arm
failures, and one control. Missing originals are not reconstructed or counted
as exact-baseline replacements.

The additional anatomical recipes cover a relaxed chin hand, back of neck,
abdomen/front chest/collar, forehead and a raised-knee fist edge. This adds
head/body surface queries and explicit elbow rules to the same fitter and
unchanged admission limits. The knee recipe uses the fist's ulnar contact edge,
not a fictitious palm contact. These new-family cases are exploratory. T029 and
T034 are additional inputs for the existing waist recipe. Recipe/core hashes
were frozen before the batch and no individual case coordinates or post-batch
recipe tuning were used.

`contact_composition.py` adds atomic two-hand chin support with separate lateral
support surfaces. Any member failure restores both arms and all fingers. It
does not validate folded-arm interlocking. The real T001 attempt fails on the
first hand's direction/skin and is fully restored. Native fault-injection tests
also prove restoration after a second-member rejection and a final-checker
exception, when earlier members have already changed bones.

`run_arm_contact_batch.py`, `verify_arm_contact_batch.py`,
`render_arm_contact_batch.py` and `report_arm_contact_batch.py` create the
separate `outputs/standin-arm-refine-20261029/contact-batch-v1/` package.
All 13 native results were reimported: **six applied, seven restored**. Applied
results are the three repeats plus T029/T034 waist hands and T025's collar-area
hand. The repeated three poses have exactly the same world bone matrices as the
previous accepted exports. T029/T034 each change 85 to 95/100 in the AI arm-only
reassessment. T025 passes geometry, but its right-hand direction/two-hand relation
remains uncertain: 55–80 becomes 55–90/100, with no confirmed point recovery.
All other judgments remain unchanged.

Failures are T001 (paired chin), T002 (neck), T003 (relaxed chin), T018 (knee
reach), T019 (chest), T024 (abdomen) and T027 (forehead control). Their actual
failure receipts and separately labelled diagnostic candidates remain visible.
All final failed FBXs equal the source bytes. A redundant FBX re-export had
introduced a 2.09e-5 matrix difference despite a perfect in-memory rollback;
rejected operations now copy the immutable original instead of requantizing it.
The verifier retains the redundant historical exports separately and checks the
exact original fallback. It does not loosen the protected-bone tolerance.

The report contains 248 verified native 640px views, filters for applied/restored
and new/repeated cases, and a toggle that clearly labels rejected diagnostics.
It inventories the 12 exclusions and the extra control, with 25 baseline/after
ledger cases (50 schema-valid records) scored by the unchanged frozen v3.2
scorer. No average combines upper/full scopes; no overall accuracy claim is
made. Source/image/camera/code hashes, static links and JavaScript syntax pass.
UI interaction is still not_run under the file-URL policy. No evaluation,
original library, production configuration, shared schemas or service changed.

User reference corrections after the batch are recorded separately in
`intent-corrections.json`; raw requests, receipts and native FBXs are preserved.
T002 is an umbrella-shaft grip with the other hand holding a phone, not a
back-of-neck contact. Its neck trial is invalid for the intended task, so the
effective report counts six applied, six fitting rejections and one wrong-target
trial (seven source-retained files physically). T018 requires the screen-right
hand on the raised knee and the opposite hand lower. Its previous one-arm reach
rejection does not evaluate that complete asymmetric pose. Neither corrected
task has been rerun. The UI now says modification not applied / existing result
retained, explicitly stating that pre-existing arm errors remain.

## Per-arm semantic retry and v4 native stage

The subsequent `semantic-retry-v1/run06` retries T002 and T018 with **both arms**.
`relational_parts.plan_arms` selects separate object/support/free-space roles
from explicit anatomical-left/right Korean clauses; it rejects absent sides,
ambiguous actions, contradictory side wording and negation. It is a bounded
language-rule selector, not image classification. The saved COCO17 evidence in
both cases has all-zero confidence and is excluded from geometric targets.
No frozen grades are used for selection. These corrected development cases do
not measure general recognition accuracy.

`relational_adapter.apply_relational` fits fixed-length arms to current 3D body
landmarks, current raised-knee skin and per-role hand frames. Umbrella and phone
receive distinct `grip` / `hold_phone` shapes; the opposite lowered knee-case
hand receives a fist. Actual full skin inspection, 65° swing / 35° twist / 165°
elbow limits, 2e-5 length/goal/protected thresholds and atomic rollback remain.
Four native regressions prove whole-pose rollback on second-member failure,
checker exception, final rejection when the support leg changes, and refusal to
export unbaked non-arm pose deformation before any mesh mutation.

`v4_adapter.apply_arm_stage(arm, meshes, clauses=..., skin_check=...,
rig_profile=..., skeleton_scores=...)` returns accepted/reasons, selected parts,
changed joints and immutable before/after world matrices. `check_arm_stage`
rebuilds the current knee surface and rechecks the combined state. V4 may run a
leg stage first, then fit arms to that final leg pose; it must independently
recheck/export the result. No production defaults or shared API files change.

`native_export.bake_arm_rest` applies the rest pose only to arm/finger bones.
Full-rest rebaking introduced 2.0869e-5 unrelated matrix drift on T018; the
selected-only export passes the original 2e-5 limit (T002 5.90e-6, T018 1.45e-5)
after actual FBX reimport, with no new skin warnings. No limit is relaxed.
Run06 preserves exact source snapshots/hashes; later language admission guards
do not overwrite its historical inputs. Procedural umbrella/phone proxies are
additional review artifacts. Their sampling checks are not complete prop/skin
triangle collision certification.

The report exposes each arm's description, selected part, target, hand preset,
changed chain and native skeleton overlay, plus 44 shared-camera 640px images.
AI arm-only reassessment with the unchanged frozen scorer is T002 0–20→20–40/45
and T018 55–75→65–95/100. T018 knee contact is 4.97mm median / 6.95mm p90, but
its reference forearm direction remains uncertain; the upper score bound is not
confirmed recovery. The lowered opposite fist matches. Non-arm judgments and
the original evaluation are untouched. Published controls are exercised in a
Node VM over 32 states / 128 local image targets and native skeleton overlays.

The user's hand-direction correction is saved separately in
`orientation-requests.json` and run09. The umbrella width axis now respects
left/right handedness: the thumb side points up, rather than reversing the left
hand. Native admission measures thumb-side up >= .6 and thumb above pinky >=
.03m. The lowered fist's dorsal normal must point up >= .6; its candidate wrist
positions are closer to the same-side knee. The language selector distinguishes
"back of hand up near knee" from "hand on top of knee" using bound knee phrases.
Run09 reimport proves corrected thumb up +.744 (previous -.744), dorsal up +.873
(previous -.163), fist-to-knee distance .405→.375m, and exactly unchanged opposite
arm matrices in both cases. Reversed directions are explicitly rejected. Both
saved native cases retain the original safety thresholds and have no new skin
warnings. The report adds hand detail and direction evidence (48 native views).
Prior run06 report/grades remain in `*-before-direction-correction.*` files;
the current AI score statuses are unchanged and T018 knee-arm uncertainty remains.

## Language-derived directions across the remaining eleven cuts

`language_plan.plan_language_arms` accepts separately sourced left/right natural
language descriptions and an explicit trial-side scope. It infers supported
directions from the action: umbrella thumb side up, the knee-side lowered fist's
dorsal side up, palm toward a head/body support, waist fingers down, or a confirmed
upward offered palm. Each condition records explicit-language versus action-rule
inference. Unsupported explicit overrides, contradictions and occluded targets
hold before mutation. These are bounded conventions, not universal pose facts.
The reference descriptions were produced by AI visual reading of the original
images; this is not an end-to-end automatic image-recognition benchmark. Active
sides retain the previous offline experiment scope, rather than claiming that
evaluation labels were used to detect a new refinement region.

`language_adapter.apply_language_plan` applies those plans transactionally. It
reuses the native contact fitter with a global `[0,.75,-.75]` elbow-pole candidate
set (optional `pole_bias=0` preserves existing behavior), verifies actual inferred
directions after each admitted contact, and rejects a good contact with a reversed
finger direction. Paired chin support tries the same `.26/.34` lateral fractions
for every pair and rolls back both arms. New free-space `palm_up_offer` uses a
cupped hand in front of the torso center and explicitly checks the palm normal.
`native_export` can bake selected sides only; default remains both arms.

`outputs/standin-arm-refine-20261029/language-batch-v2/run03` reruns all thirteen
native cuts (remaining eleven plus two development repeats), with immutable
reference descriptions, source snapshots and no score data in selection.
Eight results pass saved-FBX admission; four contact/wrist failures return exact
source bytes; one is held for occlusion. Of the remaining eleven, six apply,
four fail admission and one is held. T019 is corrected from assumed chest contact
to an upward offered palm. T025's cloak-obscured hands invalidate the earlier
collar-contact assumption, so that candidate and its possible score recovery are
withdrawn. T001/T003 fail contact direction/skin, T024 skin, T027 wrist twist.
No safety threshold is loosened.

AI selected-arm reassessment with the unchanged frozen v3.2 scorer adds T019's
confirmed +10 (55–80→65–90/100). T025's old uncertain upper-bound increase is
removed. Versus original, the five upper-body cases average25–29→31–35/45;
the eight full-body cases average71.875–80.625→79.375–89.375/100. Versus the latest
previous arm results, paired total change is0 for upper-body and0–10 for full-body.
Scopes are never mixed; the knee-arm uncertainty remains. The package includes
320 actual640px renders with common cameras for original/previous/current,
39 schema-valid reviews,32 published-control states /1248 image and skeleton
targets, and five native regressions: two post-mutation checker failures restore
every matrix, reversed waist fingers and offered palm fail, and occlusion holds
without mutation. Original evaluation and previous reviews remain hash-identical.

## Current-state follow-up and actual fitted-part thumbnails

`outputs/standin-arm-refine-20261029/arm-review-v3/run02` starts from the accepted
`language-batch-v2/run03` FBX, rather than the original evaluation pose. It adds
T024 right palm/temple contact, T026 left arm behind the lower back while keeping
the right chin support, and T027 left palm on the front forehead with the dorsal
normal forward. User corrections are explicitly distinguished from the initial
AI visual descriptions. These three are follow-ups; the other ten are reused,
not claimed as newly executed trials.

Forearm axial search now spans a full circle relative to the input's arbitrarily
baked rest frame. This is a candidate-search correction, not a relaxed anatomical
admission angle: canonical-master wrist swing65/twist35, actual skin, fixed bone
lengths and protected-chain limits are unchanged. Forehead dorsal-forward is both
inferred from the supported contact convention and measured in the head frame.
Explicit contradictory directions hold. `behind_back` is a separate role with
a horizontal hand across the lower back and an independently measured wrist
behind the torso; a forward hand cannot pass by having a plausible elbow alone.

All three saved FBX files pass reimport checks: temple gap4.65mm median,
forehead gap2.58mm median, and behind-back wrist13.35cm behind the spine. New or
worsened skin warnings are zero; existing unrelated warnings and within-hand
finger collision limitations remain. Four native regressions reject a reversed
forehead hand and a forward behind-back hand, and prove post-mutation exceptions
restore the current input exactly for both contact and relational fits.

The new report displays actual fitted procedural arm surfaces from the final
native FBX, with recipe ID, anatomical side, hand preset, target, and rejected or
preserved status. Thirteen applied arms have52 native640px views; these are
result-surface extractions, not independent fixed source meshes. Phone evidence
is explicitly `phone_hold`/`hold_phone`: holding the object exists, head/gaze
alignment to look at the screen does not. The thirteen-cut comparison reuses
ten prior results and scores only the selected three follow-up arms. Those arms
were already categorical matches: scores remain T02465/100,T02640/45,T02740/45.
This contact/detail improvement is not presented as new score recovery or an
overall accuracy claim. Published JS is exercised in32 states,1664 image targets,
1248 skeleton overlays, and all five filters.
