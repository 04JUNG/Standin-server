# P0·P1 코드 리뷰 — 2026-10-16

## 범위와 결론

배포 중인 inference 기준 `c83805125a37dd88088bd302f22b1488dbcfd1e1`에서 분리한 PR이다. P0 최신 20종 자산 명세와 P1 근거 기반 자동 적용 정책을 검토하고, 정상 입력 보류 및 BFF 기본값 회귀를 수정했다. 아래 재현 케이스의 문제는 수정·회귀 검증했다. 스테이징 배포와 merge는 검토 후 별도 진행한다.

## 리뷰에서 수정한 문제

| 우선순위 | 재현 조건과 영향 | 수정 및 회귀 증거 |
|---|---|---|
| P1 | 머리/다리 비율 없는 반신에서 명확한 프레임과 팔 볼륨까지 같은 윤곽 1표로 묶여 항상 보류 | v5 `evidence_regions`로 같은 부위의 상관 라벨은 묶고, 명시적으로 서로 다른 부위의 프레임/볼륨은 독립 근거로 인정. `test_clear_upper_body_frame_and_arm_volume_are_independent`에서 반신 허용·같은 torso 근거 보류를 함께 검증 |
| P1 | strong presentation만 남은 여성형 입력에서 기존 `auto_presentation_default`를 없애면 남성 catalog 기본값으로 복귀 가능 | 소유권·provider·명확한 호환 계열·일치하는 자산 조건에서 기존 출처 보존. `decision=presentation_default`로 체형 판독과 구분. 실제 BFF fixture에서 여성형 유지 검증 |
| P1 | 서로 다른 부위의 visible coverage와 outline representation을 결합하면 스틱 팔/숨은 몸통이 체형 근거가 됨 | 같은 부위에 두 조건이 모두 있어야 사용. 교차 부위 근거 차용 회귀 검증 |
| P2 | 머리와 다리는 보이는 전신에서 팔 윤곽 누락만으로 등신 근거가 삭제됨 | 등신의 필수 조건을 head/legs·전신·단축 없음으로 제한. 스틱 몸통 등 기존 배제 조건 유지 |
| P2 | 추가 투영 가설이 동일 체형들을 동점 지지하는 것만으로 명확한 전체 1위를 보류 | 각 가설의 최선 후보 집합에 전체 1위가 포함되면 안정적. 실제 모순 가설은 계속 보류 |

API는 decision/policy/suggestion을 typed 필드로 문서화하고, 허용 여부·출처·버전·ID의 모순만 체형 sidecar unavailable로 축소한다. 기존 people/Top-K는 유지한다. decision 없는 기존 body-match.v1 응답도 허용한다.

## P0: 최신 20종 명세

정본은 `config/body_presets.v2.json`이다. 남성 0.6.12 9종, 여성 0.2.3 11종이며 컨트롤러의 literal PRESETS에서 읽는다. 모델·컨트롤러·neutral FBX·authoring QA의 실제 SHA256와 QA source identity를 검증한다. controller를 import/실행하거나 Blender를 호출하지 않는다.

| 원본 | preset | 원본 표시명 | 새 분석용 body_id |
|---|---|---|---|
| male | BASE | 기본 체형 | `male-base-v0-6-12` |
| male | IDEAL83 | 8.3등신 · 장신 미형 | `male-ideal83-v0-6-12` |
| male | IDEAL70 | 7등신 · 균형 미형 | `male-ideal70-v0-6-12` |
| male | TOON64 | 6.4등신 · 가녀린 만화체 | `male-toon64-v0-6-12` |
| male | MUSCULAR | 근육형 · 완만한 가슴 | `male-muscular-v0-6-12` |
| male | MUSCLE_MAX | 근육 MAX 2.0 | `male-muscle-max-v0-6-12` |
| male | BULK | 벌크형 | `male-bulk-v0-6-12` |
| male | SOFT | 소프트 · 부드러운 연결 | `male-soft-v0-6-12` |
| male | FAT_MAX | FAT MAX 1.7 | `male-fat-max-v0-6-12` |
| female | BASE | 여성 · 기본 | `female-base-v0-2-3` |
| female | SLIM | 여성 · 마른형 | `female-slim-v0-2-3` |
| female | IDEAL83 | 8.3등신 · 장신 미형 | `female-ideal83-v0-2-3` |
| female | IDEAL70 | 7등신 · 균형 미형 | `female-ideal70-v0-2-3` |
| female | TOON64 | 6.4등신 · 성인 만화체 | `female-toon64-v0-2-3` |
| female | MUSCULAR | 여성 · 운동형 | `female-muscular-v0-2-3` |
| female | MUSCLE_MAX | 여성 · 근육 MAX | `female-muscle-max-v0-2-3` |
| female | SOFT | 여성 · 소프트 | `female-soft-v0-2-3` |
| female | FAT_MAX | 여성 · FAT MAX 1.4 | `female-fat-max-v0-2-3` |
| female | NEW_FAT | new_fat | `female-new-fat-v0-2-3` |
| female | NEW_MUSCULAR | new_근육형 | `female-new-muscular-v0-2-3` |

TOON64는 6.4등신 표현이며 청소년/아동 연령으로 매핑하지 않는다. head bucket h6 등 기존 관측 어휘는 정확한 실측이나 나이가 아니다. 폐기된 teen/child ID를 기존 작업에서 새 자산으로 자동 치환하지 않는다.

명세의 override는 메시 기본값과 결합한 최종 파라미터가 아니다. 실측·투영·제품 포즈별 QA·converter pin·runtime 승격은 pending이다. authoring neutral QA 통과를 제품 사용 승인으로 바꾸지 않는다. 모든 자산은 draft, supported_pose_ids는 비어 있다.

로컬 원본을 가진 작업자는 아래처럼 bundle을 재생성한다. 원본 경로는 해당 작업자의 repository root를 지정한다. Git에는 FBX·원본 이미지·AWS snapshot·키를 넣지 않는다.

```bash
python scripts/build_body_catalog_v2.py \
  --delivery artifacts/body-contact-upgrade-2026-10-16/DELIVERY.json \
  --repository-root "$PWD" \
  --out /tmp/body-p0p1-draft
```

이번 실제 원본 재검증의 canonical manifest SHA256:
`bf6863da6c8dec31281b8af2949ee3acc8efa1e6d05f9d690bcae0315f62b83b`.

bundle의 inference 20종도 전부 draft이며 BFF assets=[], converter characters={}다. 세 소비자 파일은 동일 source_manifest_sha256를 공유한다. 기존 runtime catalog나 기본 환경변수를 바꾸지 않는다. 원본 파일 없는 CI에서는 합성 source/hash/QA fixture로 생성기 계약을 검증한다.

## P1: 적용과 제안의 구분

`application_policy.allowed`는 보정된 확률이 아니다. 최소 2개 독립적인 명확한 형태 근거, 전체 순위와 명확한 근거 순위의 동일한 유일 1위, 투영 가설 간 모순 없음이 필요하다. uncertain 단서만으로 동점을 깨거나 순위 0점을 받았다는 이유로 적용하지 않는다.

스틱은 체형·관절 비율 근거에서 제외하고, 매끈한 표면의 근육선 생략을 muscle=0으로 읽지 않는다. 잘림/단축의 등신과 숨은 볼륨을 배제한다. 보류 시 auto_body_id/selected_asset/applied_body_id=null, selection_state=unavailable, render_required=false다. suggested_body_id는 검토용이며 기존 BFF 자동 추천 입력으로 사용하지 않는다.

명확한 presentation만 있는 기존 기본값 경로는 `application_policy.allowed=false`, `decision=presentation_default`, `selection_source=auto_presentation_default`로 유지한다. hair/costume만의 약한 단서·provider 실패·소유권 불명은 이 예외를 통과하지 않는다.

## BFF 경계 검증

`tests/fixtures/body_bff_v1/`는 실제 스테이징 BFF `04JUNG/Standin-app-server@ee0ba879cadb4e1bc71cedbb5640d1671ff049c4`의 model.ts/recommendation.ts를 TypeScript 5.9.3으로 ES2022에 transpile한 고정본이다. 원본·컴파일 결과 hash는 provenance.json에 기록한다. BFF 로직을 Python으로 재작성하지 않는다.

`test_body_bff_contract.py`는 inference service → `/analyze` HTTP 직렬화 → 이 JS mapper/resolver를 통과한다. 정상 자동 선택, 보류 후 사용자/카탈로그 기본값, 수동 선택과 revision, fixed_default, 명시적 auto 오류, 명확한 여성형 기본값, hair-only 보류, shadow를 검증한다. CI는 Node 22를 설치하며 이 검증을 생략하지 않는다.

이는 실제 BFF resolver 계약 검증이다. BFF DB 저장·UI·converter 실제 렌더 전체를 실행한 E2E는 아니다. 해당 저장 로직과 BFF 배포에는 변경이 없다.

## 검증 결과

최종 로컬 회귀: **824 passed, 8 skipped, 28 subtests passed**. skip은 포즈 DB/BVH·선택형 capture/평가 라벨 자료 부재이며 체형/BFF 테스트는 전부 실행했다. 독립 smoke **54/54**, compileall, API schema/example 생성도 통과했다. 기존 Starlette/httpx 및 tar 추출의 deprecation warning 2개가 있다. 로컬 Docker daemon 부재로 이미지 빌드는 실행하지 못했으며 PR CI에서 확인한다. 재현 명령은 다음과 같다.

```bash
python -m pytest tests/ --ignore=tests/converter -q -rs
python tests/test_smoke.py
python -m compileall -q src api scripts pose_curation
python scripts/export_body_api_contract.py
```

실제 원본 20종 hash/identity 검증과 생성 manifest 동일성도 확인했다. 테스트는 합성·기록된 관측 및 BFF 코드 계약을 검증하며, 새 prompt의 실제 작가 러프 정확도나 운영 허용률 보정을 주장하지 않는다. 실측·투영·독립 실이미지 평가는 후속 검증이다.
