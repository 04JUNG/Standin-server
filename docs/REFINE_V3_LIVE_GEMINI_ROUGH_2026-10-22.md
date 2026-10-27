# 실제 러프 · Gemini · FBX 검증 — 2026-10-22

후속: [2026-10-23 실제 적용 결과](REFINE_ACTUAL_ROUGH_APPLIED_2026-10-23.md)에서
동일 러프의 파생 체형·손 1개 조합이 최종 FBX 안전 검증을 통과했다.
아래는 수정 전 실행의 실패 기록이다.

실제 러프 3장을 **Gemini 3.8 Flash + Human-Art + 전체 라이브러리 검색**으로
새로 실행했다. 실호출에서 v3 provider 스키마 오류 2건을 찾아 수정했다.
수정 후 분석 요청은 3/3 정상 응답했지만, **이번 실제 러프에서 안전 승인된
최종 FBX는 0건**이다. 앞서 통과한 명시적 중립 QA 스냅샷 1건과 구분한다.
운영 배포·플래그 활성화·merge·안전 profile 등록은 하지 않았다.

## 입력과 실행 범위

- 원본 `in/124629.png`: 단독 전신 러프, 한 손을 골반 부근에 둔 자세.
- 원본 `in/131040.png`: 3명과 가림/상호작용이 있는 러프.
- 원본 `in/131224.png`: 오토바이에 가려진 인물이 있는 러프.
- 이미지 SHA, 라이브러리 DB SHA, 모델/decoder/calibration manifest, 실제 응답 모델,
  호출 종류·오류·토큰·시간·source BVH·손 프리셋·메시 검증 SHA를 기록했다.
- pinned Human-Art M ONNX/rtmlib/onnxruntime과 실제 5,795개 포즈 DB 사용.
  검색 카메라/정책에서 허용된 포즈는 5,784개다. DB와 metadata는 새 임시 디렉터리에
  복사했고, 원본 이미지·BVH·모델·릴리즈 FBX·`.env`는 수정하지 않았다.
- Google API key는 기존 `.env`에서 메모리로만 읽었다. 다른 `.env` 설정과 운영 URL은
  불러오지 않았다. 로컬 Converter는 `127.0.0.1`에만 열고 테스트 후 종료했다.
- 본 실험은 분석·기존 person tags/자세 설명/독립 좌우 검토/손 판독/조건부 관절 보완,
  검색, 자격 있는 선택 후보의 F25 refine, 손 메시 screen과 별도 strict FBX QA 범위다.
  포즈 생성 fallback, 체형 분류기, BFF, 업로드, preview는 포함하지 않는다.

## 실호출에서 발견하고 수정한 문제

1. v3 evidence를 켜면 분석 3장 모두 **HTTP 400 INVALID_ARGUMENT**였다.
   중첩 배열/숫자 bounds가 포함된 optional evidence grammar를 단순화한
   differential probe는 HTTP 200을 받았고, 실제 이미지 요청도 정상 동작했다.
   optional evidence의 provider grammar에서만 decoding bounds를 제거했다.
   기존 core schema/required 필드와 서버 `MAX_BYTES=96000`, `MAX_PEOPLE=20`,
   각 그룹 상한 16/8/6/8, exact ID·scope·enum·field 검사 및 권한은 유지했다.
   wire schema version은 `refine-evidence-schema-v2`다.
   [Google 공식 문서](https://ai.google.dev/gemini-api/docs/structured-output#limitations)도
   복잡한 중첩 schema의 400 가능성과 application validation 필요성을 설명한다.
2. 실제 손 응답은 schema가 허용한 `right_hand`/`left_foot`를 contact effector로
   냈지만, strict parser는 palm/back/sole 등 더 정확한 부위만 허용해 그룹을 폐기했다.
   provider enum을 **기존 parser의 허용 부위와 일치**시켰다. parser 허용 범위는
   늘리지 않았다. 같은 이미지의 추가 손 실호출 1회에서 `status=valid`를 확인했다.
   애매한 sole 좌우는 parser가 그대로 `evidence=unknown`으로 낮췄다.

두 probe는 이미지 없는 grammar serving 확인이며 max output을 작게 제한해 응답이
잘렸다. 완성된 분석/품질 성공으로 세지 않는다. 처음 400과 timeout 기록도 보존했다.

## 수정 후 본 실험 결과

각 이미지를 v3 evidence/shadow, off 순서로 새로 호출했다. 같은 provider 응답을
재생한 비교가 아니므로 이미지별 후보 차이를 v3의 인과 효과로 해석할 수 없다.

| 실제 러프 | v3 analyze / provider 요청 | off analyze / provider 요청 | 결과 |
|---|---:|---:|---|
| 124629 | 76.06초 / 4회 | 67.40초 / 4회 | 각 1명, 검색 후보 5개, 실제 손 메시 screen 각 5건 clear |
| 131040 | 36.61초 / 4회 | 31.77초 / 4회 | 각 3명, 기존 entangled-set 정책으로 검색 차단, 손 504 fallback |
| 131224 | 47.93초 / 6회 | 45.25초 / 6회 | 각 1명, 가림 관절 보완 후 검색 무결과, 생성 fallback 제외 |

- 본 실험 **6회 / 실제 provider 요청 28회**. 분석 6회, tags 6회, describe 4회,
  hands 6회, 조건부 skeleton review/repair 4회, 독립 좌우 검토 SDK 호출 2회다.
  원시 계측에서 마지막 종류는 `other`로 보존했다. 성공 응답이 보고한 total token
  합계 **106,102**이며, 실패한 손 요청 3회의 token/과금은 unknown이다.
- 두 경로 모두 기존 호출 종류/횟수는 14회씩이었다. 그러나 성공한 분석 3회의
  prompt token은 v3 **8,924**, off **5,591**로 3,333 token(약 59.6%) 증가했다.
  추가 호출이 없다는 것과 추가 token/지연이 없다는 것은 다르다. 분석 3회의 시간은
  각각 10.41/12.39/8.31초와 4.89/9.45/6.79초이며 통계적 비악화를 입증하지 않는다.
- 최초 진단 18회 + 소형 grammar probe 2회 + contact enum 재검증 1회를 포함하면
  이번 작업의 실제 Gemini 요청은 **49회**다. provider cost는 계산하지 않았다.
- 본 실험 손 요청 중 3회가 HTTP 504였다. 성공할 때까지 무한 재호출하지 않았고
  기존 기본 손 fallback/unknown 처리와 failure trace를 유지했다.
- 실제 손 메시 screen은 릴리즈 female BASE, frozen Converter, native shape,
  실제 Gemini 손 프리셋을 사용했다. 본 실험 10건 모두 기존 15초/호출,
  45초/session 안에서 clear였다. **opposite hands only**이며 전신 안전 인증이 아니다.
- v3 regional 권한은 5개 인물 슬롯 모두 `eligible_count=0`; 실제 /refine 진단은
  `unverified_ownership`으로 기권했다. Gemini의 visible/side 주장을 독립 관측 인증으로
  승격하지 않았다. v3가 실제 러프를 개선했다는 주장은 할 수 없다.
- 124629의 v3 live Top1은 `rokoko_Interogation_GoodCop_mixamo_00088`이고 기존 F25가
  BVH 보정을 채택했다. off live Top1은 `Female Standing Pose_01_legfix_mirror`로 달랐다.
  손 프리셋도 반복 호출에서 달랐고 confidence가 low였다. Top1/손 정확도 또는
  prompt 비악화는 이 3장으로 입증되지 않았다.
- 12초 지연 기준을 만족하지 않는다. 단독 컷의 실제 손 screen 5건이 v3 약 40.8초,
  off 약 41.7초를 썼다. 메시 validator의 약 2초와 이 end-to-end 시간을 혼동하면 안 된다.

## 실제 러프 source의 strict FBX 결과

124629에서 나온 **선택 B0와 F25 BVH**를 동일 파생 female BASE 프로필,
동일 Gemini 손 `left=relaxed, right=grip`으로 각각 Blender에 넣었다.
채널을 중립으로 바꾸거나 보정 기준을 낮추지 않았다.

| source | BVH SHA256 | strict 결과 |
|---|---|---|
| 선택 B0 | `c2234a5dc55b0eb595e4ba6aa1b5d817202cfe89e0f8d3e9a77ea83d9b62b763` | body witness에서 `discrete_shell_orientation_reversed` |
| F25 보정 | `eee645226f1f17adcad7dbe4c198c7896ba111408860a8f6d09cc5132e144169` | 같은 기준에서 차단 |

두 실행 모두 새 neutral toe repair 단계의 전체 메시 검사는 통과했지만, body에서
원본 법선/면적을 유지하는 세 corner director 방향 기준을 통과하지 못했다.
실제 native shape를 neutral에서 만든 뒤 frozen Converter로 구운 **파생 pure-LBS
프로필**의 결과다. native 비선형 pose corrective가 있는 모든 제품 출력의 접힘을
판정했다는 뜻은 아니다. 이 profile에서 원본부터 해당 기준을 위반하므로 이번 결과만으로
F25가 새 접힘을 만들었다고도 판단할 수 없다.

실패 시 `qa-failure.json`에 source/body FBX bytes, 마지막 검증 단계와 원인을 남기도록
QA 도구를 보완했다. 두 건 모두 strict final receipt를 발급하지 않았고, hands/final의
전체 메시 안전을 통과했다고 집계하지 않았다. 기존 중립 스냅샷 1건의 통과 범위는 유지된다.

## 회귀와 재현

- 추론 **2,498 passed, 10 skipped**, 28 subtests passed (23.33초).
- Converter **454 passed, 1 skipped** (5.24초). 동일 Blender 전용 실행 모듈 3개는
  CPython 수집에서 제외하며, 위 실제 Blender 실행은 별도다.
- 추가 회귀는 core schema 불변, compact provider grammar, 기존 크기/그룹 상한·권한
  유지, contact schema와 parser 허용 집합 일치를 확인한다.
- Smoke 54/54, compileall, diff-check 통과. frozen Converter 파일 변경 없음.

재현 도구: `scripts/eval_refine_v3_live_rough.py --remote --images <세 원본 이미지>
--data-root <검증된 라이브러리> --manifest <Human-Art manifest> --env-file <읽기 전용 env>
--out <새 임시 폴더> --phases evidence_shadow off --converter-url http://127.0.0.1:<port>
--character-id female-base-v0-3-4 --character-sha256 <정확한 릴리즈 SHA>`.
실제 Gemini 호출이 발생한다. local Converter 미설정은 unknown으로 처리되며 조용히 건너뛰지 않는다.

[작은 검증 요약 JSON](../evaluation/refine_v3/actual_gemini_rough_pilot_2026-10-22.json).
원시 provider 응답·관절·BVH는 `/private/tmp/standin-refine-v3-live-rough-fixed-20261022`,
최초 오류는 `/private/tmp/standin-refine-v3-live-rough-20261022`에 보존했다.
실제 이미지·FBX·API key는 커밋하지 않았다.

이전부터 검토된 3장에 대한 integration pilot이며 독립 holdout, 2D/3D/손 정답,
실제 v3 품질 개선·일반 체형/손 안전·운영 SLO 인증으로 확대하지 않는다.
