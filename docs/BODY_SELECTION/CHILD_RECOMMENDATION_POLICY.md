# 아동 자산 자동 추천 정책

2026-10-20 · `body-child-recommendation.v1.manual-only`

## 기준과 허용 조건

아동 모델은 삭제하거나 retired로 바꾸지 않는다. 현재 이미지 관측 계약에는 검증된 아동 디자인 판정이 없으므로 이미지에서 **아동 자산을 자동 추천하지 않는다**. VLM이 나이를 확신한다고 쓰거나, 명확해 보이는 아동 디자인의 짧은 비율을 반환하더라도 명시적 사용자 선택이 필요하다. 이는 실제 나이 판정이 아니라 제품의 모델 선택 정책이다.

다음은 아동 자산 선택의 충분한 근거가 아니다: 큰 눈·둥근 얼굴·매끈한 피부·근육선 생략·큰 머리·짧은 팔다리·6.4등신 스타일·작은 프레임·스틱·웅크림/쪼그림·원근 단축·헐렁한 옷·가림·VLM의 clear 표시. 이러한 정보의 조합도 현재 계약에서는 아동 자동 추천 권한을 만들지 않는다. 높은 나이 확률 임계값을 임의로 붙이지 않는다.

허용 경로는 명시적인 수동 선택, 사용자가 저장한 fixed_default/user default, 수동 선택에서만 기록되는 last_used다. 이 경로의 아동 모델·제어값·카탈로그 적격성은 유지한다. 향후 이미지 자동 아동 추천을 지원하려면 별도의 검증된 디자인 증거 계약과 승인된 평가가 필요하며, 현재 clear/visible 신호를 그 계약으로 간주하지 않는다.

## 최신 코드의 경로와 확인된 허점

기준은 추론 develop `d934238fb21c4798e242ae61a5c46b19b0905aac`, BFF `262378351a2a8b1e71f431251540e81082e5d043` (작업 시작 ef7ebcac, PR 전 새 관리자 설정 머지 반영), client 읽기 점검 `9f7f024a96b20e1c6bbaf5c17cf87cda317b4b42`다. 작업 시작 시 추론 기준은 93781fb였지만 실제 수정 직전 새 develop 머지를 확인하여 갱신했다. 공용 로컬의 대량 미커밋 변경이나 옛 PR90 브랜치를 기준으로 재사용하지 않았다.

| 단계 | 최신 원격 동작 / 확인 |
|---|---|
| 최초 통합 VLM | 나이/age 필드 없음. 사람 수·박스와 독립적인 성별 디자인·근육·체격·비율만 관측한다. 프롬프트도 실제 나이 추론을 금지한다. 이번에 프롬프트·제어 스키마는 바꾸지 않았다. |
| coarse/integrated | 20종 성인/스타일 자산에 매핑. compact→TOON64는 아동이 아니다. 현행 카탈로그에서 아동을 추천하지 않는다. |
| legacy 형태 랭커 | 사용자 정의 child 자산이 들어오면 약한 등신·small_frame만으로 suggested_body_id에 child를 남길 수 있었다. 기존 회귀 테스트도 이 동작을 허용했다. 충분해 보이는 여러 형태 축은 자동 적용까지 허용할 수 있어 아동 전용 차단이 필요했다. |
| BFF | 자산·사람·포즈 조인만 맞으면 옛 child 자동 추천도 수용하고, 이전 자동 선택은 자산이 남아 있는 한 재사용할 수 있었다. 추천/선택/과거 결과 재검증에서 차단한다. |
| client | 표시용 child 분류는 카탈로그 ID/이름을 분류한다. 이미지에서 나이를 판정하는 경로는 찾지 못했다. mock/과거 선택·last_used·default는 별도 경로다. client는 수정하지 않았다. |

첨부 러프의 실제 Job ID/원격 원문/당시 선택 provenance는 제공되지 않았다. 따라서 이 이미지가 실제 운영의 어느 분기에서 child가 됐는지 단정하지 않는다. 위 허점은 최신 코드와 기존 테스트로 재현한 경로다.

## 구현

- `child_policy.py`는 카탈로그 metadata.age_design 또는 child/kid/baby/infant/toddler 자산 식별자를 판별한다. 이미지 이름/인물 ID/얼굴/등신/성별로 분기하지 않는다. 일반 small_frame/TOON64는 제외하지 않는다.
- legacy 랭커에서 child를 순위·제안·자동 선택 후보 모두에서 제거한다. 아동 기본값이 남아 있으면 기존 ordinary 자산을 기본 후보로 사용하고, 대체 자산이 없으면 선택을 요구한다. 전체 카탈로그/수동 eligibility는 그대로다.
- 제외 시 `child_recommendation_policy={version,mode:manual_only,excluded_body_ids}`와 `child_requires_explicit_user_selection`을 진단한다. 등수·rank_score는 나이 확률이 아니다.
- BFF는 자동 추천을 unavailable/body=null로 바꾸어 기존 사용자/성인 기본값 경로에 연결한다. explicit auto는 조용히 기본값으로 바꾸지 않는다. 아동 catalog_default는 새 자동 fallback으로 쓰지 않으며, 다른 안전한 기본값도 없으면 선택을 요구한다.
- 저장된 auto_recommendation/catalog_default_fallback 아동은 GET 재검증에서 resolvedBody를 지우고 needs_selection을 반환한다. DB·revision은 변경하지 않는다. manual/fixed_default/last_used/user_default_fallback은 보존한다.

## 검증과 범위

일반 성인·스타일화·웅크림·가림·소유권 불명·아동처럼 보이는 clear 관측을 포함해 아동 자동 제안 제외, 아동만 있는 카탈로그, 별칭/metadata 판별, 아동 수동 적격성 유지, adult BASE/TOON64 비퇴행을 검증한다. BFF는 구형 sidecar, 캐시된 추천, 수동/기본값/last-used, 과거 결과 provenance·읽기 전용 재검증을 검사한다.

첨부 764×1056 러프는 로컬에서 직접 열어 모자·얼굴 스타일화·웅크린 단축 자세·헐렁한 옷의 가림을 확인했다. 동일 단서를 가진 수동 구성 관측으로 정책을 재생하면, 아동 catalog_default가 남은 이전 정책은 child를 제안했고 새 정책은 adult-base를 제안했다. **새 VLM 판독·실제 운영 결과 재현·정확도 측정이 아니다.** 원본 이미지·키·원문 응답은 Git에 넣지 않는다. 상세 로컬 산출물은 `/private/tmp/standin-child-policy-evidence/ATTACHED_POLICY_REPLAY.json`이다.

기존 17컷/28명 QA는 저장된 최초/재검토 응답만 재생한다. 이 작업은 원격 VLM을 새로 실행하거나 운영 정확도·지연을 측정하지 않는다. 검증 결과는 추론 2,194 passed / 10 skipped / 28 subtests, smoke 54/54이며 BFF는 별도 PR에서 최신 develop 기준 검증한다. 17컷 사람/박스/관절/포즈 JSON과 36명 추천은 이전 V7 재생과 완전 일치했다. 승인된 28명은 남녀 28/28, 세부 프리셋 24/28을 유지한다. PR 링크는 PR 본문에 기록한다. 배포·환경 설정·자동 병합은 하지 않는다.
