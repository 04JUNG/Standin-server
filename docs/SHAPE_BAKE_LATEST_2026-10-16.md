# 체형 수치 굽기 · 최신 모델 연동 (2026-10-16)

목표: 체형 더보기에서 슬라이더로 조절한 값이 **최종 FBX와 저장 결과 포즈에 그대로 들어간다.**
대상 모델: 남성 **0.7.1**, 여성 **0.3.3** (`artifacts/body-production-release-2026-10-16-v0.7.1-v0.3.3`).
여성은 작업 도중 0.3.2 → 0.3.3(골반 보정)으로 올라가 다시 등록·검증했다. 코드 변경 없이 릴리스 데이터만 다시 만들었다.
이후 확정 모델 **남성 0.7.2 · 여성 0.3.4**(S3 스테이징 `confirmed-20261016-m072-f034`)도 코드 변경 없이 등록·검증했다(아래 절).

## 결론

- **FBX·최종 결과물 = 슬라이더 값.** converter가 승인된 원본 컨트롤러(`standin_body.py`)를
  Blender에서 직접 실행해 수치를 굽고, V3.2.5 solver → 원본 리그 이식 → 컨트롤러의 자세·접촉 보정
  → 최종 FBX를 만든다. 검토 PNG/GLB도 그 FBX에서 나온다. 컨트롤러가 바뀌어도 코드는 그대로다.
- **모델 버전 승인은 데이터다(`shape-bake.v2`).** 하드코딩된 0.6.7/0.2.2 핀을 없앴다. 새 버전은
  `scripts/build_shape_release.py`로 레지스트리·프리셋 FBX를 만들고 굽기 검증만 하면 된다.
- **버전 가드.** 카탈로그(`shapeSource`) · converter 레지스트리 · 앱 실시간 자산(SSR1/SSR2 머리말의 `source`)이
  같은 원본(컨트롤러 VERSION + .blend/.py SHA-256)을 가리킬 때만 굽고 그린다. 하나라도 다르면
  converter `409 SHAPE_SOURCE_MISMATCH`, BFF 무결성 오류, 앱은 고정 그림으로 떨어진다.

## 계약 변경 (확정)

| 위치 | 변경 |
|---|---|
| converter 레지스트리 | `shape-bake.v2`: `sources{rules → blend, controller, sha256}` + `characters{id → rules, preset, asset_sha256, defaults, controls}`. v1 거부 |
| converter `/convert-framed` | `expected_shape_source` 폼 필드 추가(shape와 함께만). 다르면 `409 SHAPE_SOURCE_MISMATCH` |
| converter 시간 | `CONVERTER_SHAPE_TIMEOUT_SECONDS`(기본 300) — 수치 굽기 전용 |
| 슬라이더 격자 | 프리셋 기본값이 반드시 격자 위(`(default-min)/step` 정수). 빌드가 `min`을 한 칸 미만 올려 맞춘다(예: head_scale .927 → min .907). ※ 운영 승급 재검사 작업은 같은 문제를 head_scale step 0.001로 풀었다 — 둘 중 하나로 합의 필요 |
| BFF manifest | asset `shapeSource{rules, blendSha256, controllerSha256}`. `shapeRuntime.format`에 v2 추가 |
| BFF 선택 응답 | 수치 선택에 `shapeSource`(현재 카탈로그에서 파생, 저장 안 함) |
| BFF 시간 | `CONVERTER_SHAPE_TIMEOUT_MS`(기본 320000) |
| 앱 | 실시간 자산 두 경로 모두 원본 일치 검사: 0.7.1/0.3.3은 정밀 이식 규칙(SSR1, 다른 세션 작업 `8fb4a83` 머지), 이식 전 새 버전은 `standin-shape-runtime.v2` 표본 자산 |

## 검증 (클라우드, Linux 2 vCPU, pip bpy 5.0.1)

> 운영 converter는 Blender **5.2.0** 고정이다. 아래는 같은 코드 경로(BlenderRunner → worker job)를
> bpy 5.0.1로 돌린 개발 검증이다. 배포 전 5.2.0(Mac 또는 컨테이너)에서
> `scripts/verify_shape_bake.py`·`inspect_shape_bake.py`를 다시 돌린다. 프리셋 FBX 해시도 Blender
> 버전마다 다르므로 운영 릴리스는 5.2.0에서 `build_shape_release.py`로 다시 만든다.

- 릴리스 빌드: 남성 9 + 여성 11 = 20개 프리셋 자산, 레지스트리 자체 검사 통과(약 1분). 같은 원본이어도 FBX 내보내기 해시는
  빌드마다 다르다(파일 시각 포함) — 레지스트리·카탈로그는 그 빌드의 해시를 함께 고정한다.
- 실제 굽기 14회(7쌍) 모두 성공, 굽기 계보(`shape_bake.controls`)가 요청 값과 일치. 아래 표(남성 기본형은 같은 0.7.1 원본의 첫 빌드 자산).
- 독립 검사(`inspect_shape_bake.py`, FBX와 컨트롤러만 사용): 최종 FBX의 관절 간 거리가 요청 값으로
  컨트롤러가 만든 골격과 0.5mm 이내, 체적 변화 부호·크기와 정점 이동 크기 상관이 중립 변화와 일치.
- 실제 BFF 코드(`convertFramed`) → converter HTTP 연동: 여성 0.3.3 풍만형(가슴 0.5·키 170·지방 1.2·지방 볼륨 1.5,
  앉은 자세) 112초에 성공, 다른 원본을 약속하면 `409 SHAPE_SOURCE_MISMATCH` → BFF `CONVERTER_REJECTED`.
- 계약 테스트: converter 190개 통과(Blender 필요한 4개 파일 제외).

| 체형 | BVH | 슬라이더 값 | 굽기(초) 프리셋/수치 | 관절 거리 오차 | 정점 이동 상관 | 체적 변화 중립→포즈 | 판정 |
|---|---|---|---|---|---|---|---|
| male-bulk-v0-7-1 | rokoko_Shamwow_Guy_mixamo_01353_mirror | 근육 1.2 · 키 186 · 어깨 0.3 | 120 / 128 | 0.002 mm (변화 최대 20.09 mm) | 0.8878 | +8.0% → +7.6% | 통과 |
| male-base-v0-7-1 | Arm Stretching_00054 | 키 190 · 등신 7.5 · 지방 0.8 · 근육 1.0 | 61 / 152 | 0.001 mm (변화 최대 33.36 mm) | 0.9699 | +82.4% → +72.9% | 통과 |
| male-base-v0-7-1 | Kneel_02 | 키 190 · 등신 7.5 · 지방 0.8 · 근육 1.0 | 79 / 86 | 0.001 mm (변화 최대 33.36 mm) | 0.8405 | +82.4% → +71.8% | 통과 |
| female-base-v0-3-3 | Arm Stretching_00054 | 가슴 0.6 · 지방 0.6 · 지방 볼륨 1.0 · 키 165 | 98 / 116 | 0.001 mm (변화 최대 25.98 mm) | 0.9838 | +18.0% → +16.5% | 통과 |
| female-base-v0-3-3 | Kneel_02 | 가슴 0.6 · 지방 0.6 · 지방 볼륨 1.0 · 키 165 | 47 / 86 | 0.001 mm (변화 최대 25.98 mm) | 0.8452 | +18.0% → +9.8% | 통과 |
| female-new-muscular-v0-3-3 | Arm Stretching_00054 | 근육 3.0 · 키 172 · 어깨 0.3 | 87 / 83 | 0.001 mm (변화 최대 7.79 mm) | 0.9465 | -14.1% → -13.0% | 통과 |
| female-new-muscular-v0-3-3 | Kneel_02 | 근육 3.0 · 키 172 · 어깨 0.3 | 118 / 64 | 0.001 mm (변화 최대 7.79 mm) | 0.7682 | -14.1% → -8.7% | 기준 미달* |

\* 여성 고근육형 무릎 꿇기: 정점 이동 크기 상관 0.77로 기준(0.8) 미달. 관절 오차 0.001mm, 평균 이동 18.9mm(중립과 같음),
체적 −14.1% → −8.7%로 슬라이더 효과는 들어갔다. 깊은 접촉 보정이 허벅지·종아리 면을 다시 밀어내서 생긴 차이로 보이며
기준을 사후에 낮추지 않고 그대로 기록한다.

관절 거리 오차는 최종 FBX의 부모-자식 관절 거리와, 같은 값으로 컨트롤러가 만든 골격의 차이다(포즈는 뼈마다
강체라 거리 = 체형 비율). 체적은 접촉 보정이 있는 깊은 자세(무릎 꿇기)에서 덜 반영되지만 방향과 크기는 유지된다.

참고: 같은 0.7.1/0.3.3 원본의 운영 승급 재검사(`artifacts/body-production-release-2026-10-16-v0.7.1-v0.3.3/REPORT.md`, Blender 5.2.0)는
converter HTTP 124/124 전달은 통과했지만 실제 포즈 표면 관통 55/100으로 **승급 보류**다. 수치 굽기 연동과 별개로 모델 접촉 품질 게이트가 남아 있다.

관찰: 여성 풍만형 + 지방 볼륨 1.5 + 앉은 자세에서 배-허벅지 접촉부에 어두운 면이 보였다(모델 접촉 보정 영역,
이번 연동과 무관한 기존 조형 한계로 보이며 모델 QA 확인 필요).

## 남성 0.7.2 · 여성 0.3.4 (확정 모델, 2026-10-16)

원본: `artifacts/body-surface-upgrade-2026-10-16/{male,female}/model` — S3 스테이징
`body-models/releases/confirmed-20261016-m072-f034-04778b9f9479`의 blend와 SHA-256이 같다
(남성 blend `9ff276f6…5998`·컨트롤러 `51baa18d…45b9`, 여성 blend `a3622a9d…ae32`·컨트롤러 `df981aed…eb1f`).
바뀐 것은 컨트롤러 `correct_pose`(표면 보존 보정, 관통 검사 끔)뿐이고 중립 체형 계산은 0.7.1/0.3.3과 같다
(40개 조합 0.0 mm, 같은 asset identity).

- 코드 변경 없음: `build_shape_release.py --source <male> --source <female>`로 레지스트리(`shape-bake.v2`)와 프리셋 FBX 20개를 만들었다.
- 실제 굽기 4회(2쌍, 무릎 꿇기) 모두 성공, 굽기 계보가 요청 값과 일치. **0.7.1/0.3.3보다 빠르다**(관통 검사가 빠짐).
- 독립 검사 2/2 통과.

| 체형 | BVH | 슬라이더 값 | 굽기(초) 프리셋/수치 | 관절 거리 오차 | 정점 이동 상관 | 체적 변화 중립→포즈 | 판정 |
|---|---|---|---|---|---|---|---|
| male-bulk-v0-7-2 | Kneel_02 | 근육 1.2 · 키 186 · 어깨 0.3 | 32 / 27 | 0.001 mm (변화 최대 20.09 mm) | 0.9264 | +8.0% → +8.4% | 통과 |
| female-base-v0-3-4 | Kneel_02 | 가슴 0.6 · 지방 0.6(physique) · 지방 볼륨 1.0 · 키 165 | 14 / 20 | 0.001 mm (변화 최대 25.98 mm) | 0.8488 | +18.0% → +15.9% | 통과 |

앱 실시간 자산(SSR1 정밀 이식, 클라이언트 `MALE_072`·`FEMALE_034` 별칭): 남성 7,091,028 B sha256 `f6d7be53…a129`,
여성 4,984,080 B sha256 `c21d170b…6f2f`. 머리말 `source`가 위 원본과 같다. 패키징 예시는 아래 "실행" 4)번.

남은 것: 확정본 S3 스테이징에는 blend·T 포즈 FBX·프리셋만 있다. 수치 굽기를 켜려면 컨트롤러 `.py`,
`shape-bakes.json`(이 converter가 쓰는 레지스트리), 실시간 자산, BFF manifest의 `shapeSource`/`shapeRuntime`이 담긴
shape-bake 릴리스를 Blender 5.2.0에서 `package_shape_staging.py`로 만들어 따로 게시해야 한다.

## 지연 시간

수치 굽기 1회 30–110초(2 vCPU). 대부분 컨트롤러 `correct_pose`(21–70초)다. 남성 BULK가 가장 느리다.
Top-K 체형 미리보기 5장은 5번 굽는다(동시 처리 1이면 수 분). 운영 전에 converter 동시성,
ALB idle timeout, 앱 대기 문구(검토 화면에 "1~2분" 안내 추가됨)를 함께 확인한다.

## 실행

```bash
# 1) 릴리스 입력 (Blender 5.2.0 권장)
blender --background --python scripts/build_shape_release.py -- \
  --source <male>/model --source <female>/model --out artifacts/shape-release-<date>
# 2) 실제 굽기 + 독립 검사
CONVERTER_SHAPE_REGISTRY=<out>/shape-bakes.local.json python scripts/verify_shape_bake.py \
  --release <out> --bvh <a.bvh> --case male-bulk-v0-7-1:muscle=1.2,height=186 --out <out>/bake-qa
CONVERTER_SHAPE_REGISTRY=<out>/shape-bakes.local.json blender --background --python \
  scripts/inspect_shape_bake.py -- --runs <out>/bake-qa/runs.json --out <out>/bake-qa/inspection.json
# 3) 앱 실시간 자산 (클라이언트 저장소 scripts/shape-runtime/build_sampled_runtime.py)
# 4) 패키지(업로드는 --upload일 때만)
python scripts/package_shape_staging.py --release <out> --bake-qa <out>/bake-qa \
  --runtime standin-body-0.7.2=<male.ssr> --runtime standin-female-body-0.3.4=<female.ssr> --version shape-<date>-v1
```

배포 순서: converter → BFF → 앱 → 인프라에서 새 manifest 선택 → `BODY_SHAPE_EDIT_ENABLED`,
`BODY_SHAPE_BAKE_ENABLED` 켜기. 이 작업은 ECS·운영 포인터·S3를 바꾸지 않았다.

## 2026-10-17 develop 통합 검증

develop `c8181b0e`의 손 라이브러리·장면 합성·palm-roll 변경을 반영했다.
`shape`/`expected_shape_source`와 `hand_left`/`hand_right`를 같은 요청 및 worker job에서
보존한다. 처리 순서는 체형 굽기 → 손 프리셋 굽기 → 프레이밍·카메라·미리보기이며,
runner는 체형과 손의 계보를 각각 검증한다. 누락·다른 수치·다른 손 프리셋은 결과를 거부한다.

- 추론 회귀 1,481개(로컬 HTTP 오류 복구 4개 포함), Converter Python 291개 통과.
  추론 9개·Converter 1개는 기존 외부 자산/선택형 환경 조건으로 skip했다.
- 신규 결합 계약 9개와 기존 shape-bake 계약 28개를 CI에 포함했다.
- 고정 Blender 5.2.0 `fbe6228777e7`에서 기존 합성 변환 회귀 28/28 통과.
- 원본 SHA가 일치하는 남성 0.7.2·여성 0.3.4 BASE 자산의 임시 복사본으로,
  합성 팔 들기 BVH + `muscle=0.5` + 왼손 fist/오른손 relaxed를 실제 worker에서
  각각 실행했다. 두 건 모두 체형 controls/digest, 손 선택·몸뼈 보존, 최종 FBX 해시 검증을
  통과했다(각 6.96초/5.46초). 원본 자산·S3·배포 설정은 변경하지 않았다.

이 검사는 두 BASE 조합의 실행 검증이며 모든 체형/포즈/손 조합의 메시 건전성,
실제 러프 품질, PNG/GLB/CSP 표시 또는 운영 부하 검증을 대체하지 않는다.
