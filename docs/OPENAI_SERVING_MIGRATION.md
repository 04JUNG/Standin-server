# 서비스 API의 OpenAI 전환

이 변경은 `/analyze`의 이미지 분석, 태그·손 모양·자세 설명, 독립 좌우 검수,
체형 판정과 검수, 스켈레톤 보완, 자연어 포즈 생성의 OpenAI 경로를 연결한다.
기존 프롬프트, 인물 대응, 검색·리파인 권한, 로컬 응답 검증은 공유한다.
전용 scene worker에도 OpenAI evidence reader를 추가한다.

**50컷 자동평가의 코드·모델·실행 설정은 이번 작업에서 변경하지 않는다.**
평가에 쓰이는 Gemini 키를 삭제하거나 평가 모델을 OpenAI로 바꾸지 않는다.
배포 workflow의 기존 평가 단계도 유지되므로, 평가 실행 정책은 해당 담당자가
별도로 관리해야 한다. 회귀 테스트에서는 배포·평가·유료 호출을 실행하지 않는다.
2026-10-31 운영 전환 시에는 각 환경 키로 합성 이미지 1건의 OpenAI 호출을
확인했고, 기존 이미지의 스테이징 inference 서비스 설정을 전환했다. 50컷은 실행하지 않았다.

## 환경별 설정

GitHub `staging`과 `beta` environment에 각각 설정한다. `develop`은 staging,
`main`은 beta 설정을 읽는다. 서비스 배포는 `--serving` 프로필을 적용하므로
변수가 비어 있어도 기존 Gemini 설정을 유지하지 않고 **OpenAI / gpt-6-luna / low**로
전환한다. 분석·체형·family review·자연어 생성·스켈레톤 검수 모델 모두 같은
프로필을 사용한다. 환경별 명시값은 프로필을 덮어쓸 수 있다. 키/ARN은 기존
환경의 참조를 그대로 사용한다. 독립 실행 평가 태스크에는 이 프로필을 적용하지 않는다.

| 변수 | 값 |
|---|---|
| `VLM_PROVIDER` | `openai` |
| `OPENAI_MODEL` | 운영자가 선택하고 해당 키로 접근 가능한 분석 모델 ID |
| `BODY_VLM_PROVIDER` | 체형 sidecar 사용 시 `openai` |
| `BODY_VLM_MODEL` | 선택한 체형 판정 모델 ID. 기존 저비용 역할을 고려해 별도 지정 |
| `POSE_GENERATION_PROVIDER` | 자연어 생성 사용 시 `openai` |
| `POSE_GENERATION_MODEL` | 선택한 자연어 생성 모델 ID |
| `SKELETON_REPAIR_MODEL` | 필요 시 별도 모델. 미설정이면 분석 모델 사용 |
| `BODY_COARSE_FAMILY_REVIEW_MODEL` | 별도 family review를 사용 중이라면 OpenAI 모델로 지정 |
| `OPENAI_REASONING_EFFORT` | 서비스 기본값 `low`. 변경 시 선택 모델이 지원하는 값만 지정 |

기본 모델은 기존 단일 세트 실험에서 JSON 응답 성공과 지연을 확인한
`gpt-6-luna`, 추론 강도는 `low`다. 해당 실험은 전체 서비스 품질 보증이 아니므로
배포 후 실제 서비스 요청을 확인한다. 모델 변경은 환경별 변수로 명시한다.
호출 1회는 기본 45초, 분석 재시도 포함 예산은 75초다. OpenAI SDK 자동 재시도는
끄고 기존 애플리케이션의 예산 내에서만 재시도한다. 인증·권한·결제 한도 오류와
timeout은 반복 재시도하지 않는다. 선택 모델의 추론 설정과 지연은 실제 소규모
서비스 요청으로 확인해야 한다.

키는 해당 환경의 Secrets Manager 시크릿에 팀 담당자가 저장한다. ECS의
`OPENAI_API_KEY` secret reference가 해당 JSON 항목(예: `openaiApiKey`)을 읽어야 한다.
키를 GitHub variables, task definition의 일반 environment 또는 저장소에 넣지 않는다.
`prepare_vlm_deployment.py`는 값 자체를 읽지 않고 참조·provider/model 일치만 검사한다.
다른 환경의 키·ARN·역할·이미지는 복사하지 않는다.

scene worker는 별도 이미지/서비스이므로 `SCENE_VLM_PROVIDER=openai`,
`SCENE_VLM_MODEL=<선택 모델>` 및 그 task의 `OPENAI_API_KEY` secret reference를
별도로 적용해야 한다. 현재 장면 모델·자산/Blender 준비 조건은 유지한다.

## 검증과 제한

OpenAI 공식 SDK를 실제로 사용하되 HTTP MockTransport 또는 로컬 가짜 HTTP
서버로 요청·오류를 검증한다. 외부 API 과금과 50컷 실행 없이 회귀 검사가 가능하다.
SDK 호출은 Responses API이며 이미지 입력, JSON 응답, 응답 ID와 실제 토큰 사용량을
기록한다. 닫힌 필수 필드 스키마는 strict Structured Outputs를 사용하고,
동적 map/선택 필드가 있는 기존 계약은 JSON mode와 원본 스키마 지시 및 기존
로컬 검증을 사용한다. 프롬프트의 선택 필드를 임의로 필수/null로 바꾸지 않는다.
`store=false`이며 원시 provider 오류 본문은 사용자에게 반환하지 않는다.

키 미등록·모델 미지정 상태에서는 실제 AI 기능이 동작할 수 없다. 가짜 분석 결과로
운영 성공을 표시하지 않는다. 이 코드의 오프라인 테스트 통과는 해당 계정의 모델
접근 권한·요금·실제 이미지 분석 품질과 응답 지연의 검증을 대신하지 않는다.

등록 후 순서는 staging의 제한된 실제 분석 요청 → 로그/지연/검색 결과 확인 →
검증된 변경을 main에 반영하고 beta의 별도 키/모델로 확인하는 것이다.
rollback은 각 환경의 provider/model 설정과 이전 immutable 이미지를 함께 복구한다.
Gemini로 되돌릴 때에는 `VLM_PROVIDER`, `BODY_VLM_PROVIDER`,
`POSE_GENERATION_PROVIDER`와 활성 모델 필드를 모두 명시해야 하며,
혼합된 provider/model은 배포 전에 검증 오류로 차단한다.

공식 API 참고: [Responses 사용](https://developers.openai.com/api/docs/quickstart),
[구조화 출력](https://developers.openai.com/api/docs/guides/structured-outputs),
[이미지 입력](https://developers.openai.com/api/docs/guides/images-vision).
