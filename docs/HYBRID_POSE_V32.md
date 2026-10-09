# Hybrid Pose v3.2 — 단일 호출 + Low 시간 최적화

이 PR은 실험에서 검수한 파이프라인을 **독립 실행 가능한 모듈/CLI**로 이식한다.
기존 `/analyze`, 3D 포즈 검색·리타게팅·배포 설정은 바꾸지 않는다.
운영 API에 연결하거나 기본 경로를 바꾸는 것은 별도 검증/PR 대상이다.

## 흐름

```text
원본 러프 → Human-Art RTMPose-M → Gemini 검토·부분 수정 통합 1회 (Low)
                               → 로컬 코드 검증 → 최종 관절 JSON + 변경 이력 + 시간
                                 pass: RTM 유지 / repair: 패치 병합
                                 review_needed: RTM 유지 + 검수 필요
```

- 평가 대상: 인물 누락, 찾을 수 있는 관절 누락, 명백한 위치/연결 오류, 가짜/중복 인물.
- 신뢰도가 낮다는 이유만으로 수정하지 않는다. 모델별 관절 임계값은 좌표 유효성 필터이고 VLM 분기 기준이 아니다.
- 미세 각도는 수정 조건이 아니다. GT/정답 인물 수를 입력하지 않는다.
- 수정은 지적된 관절과 같은 팔/다리의 필수 연결점만 허용한다.
- 수정하지 않는 좌표·상태·RTM 점수와 기존 인물 ID는 유지한다. 수정점의 RTM 점수는 제거한다.
- 새 인물은 별도 ID를 부여하고 COCO-17 항목을 검증한다. 화면 밖/근거 부족 관절은 null이다.
- 기본 모드는 컷당 API **최대 1회**, 수정 후 **VLM 재평가 없음**.
- `--vlm-mode two-stage --thinking-level default`로 기존 1~2회 방식에 명시적으로 복귀할 수 있다.
- 자동 재시도/모델 변경/추론 설정 폴백은 없다. Low 미지원 모델의 설정 오류도 그대로 반환한다.
- 오류는 `status=error, final=null`; 보존한 `rtm`을 성공 결과로 둔갑시키지 않는다.
- `status=ok`는 실행/형식 검사 완료일 뿐 사람의 품질 승인이나 각도 합격이 아니다.

## 실행

2026-10-08 사용자는 비교 결과의 육안 품질을 근거로 **Human-Art RTMPose-M을
향후 하이브리드 실험 및 독립 추출 CLI의 기본 모델로 확정**했다. 이는 전체
처리시간 개선이 입증됐다는 의미가 아니다. 같은 27컷의 기록상 평균 처리시간은
기존 RTMPose-X 기반 45.38초, Human-Art 기반 51.18초였고, 비교에는 과거 결과
재사용·런타임·임계값 차이가 포함된다. 이후 단일 호출·Low 실험 결과는 아래에
별도로 기록한다. 이 코드 반영으로 새 API 실험이나 운영 배포를 시작하지 않는다.

하이브리드 CLI의 기본 선택은 `humanart-m`이다. 기존 RTMPose-X는
`--pose-model current-x`로 명시해 사용할 수 있다. `/analyze`와 전역
`POSE_MODEL_VARIANT` 기본값은 이 변경으로 바뀌지 않는다.

Human-Art는 검증된 `--pose-manifest`(또는 `POSE_MODEL_MANIFEST`)와 승인된
`POSE_CANARY_STAGE` 설정이 필요하다. 기존 해시·RGB·SimCC·런타임 검증을
우회하지 않으며, production에서는 라이선스 승인/rollout 상태도 검사한다.
현재 저장소의 상용 라이선스 검토는 미승인이므로 **운영 전환 완료가 아니다.**
파일이 없거나 검증 실패 시 API 호출 전에 종료하며 current-X/mock으로 대체하지 않는다.
앞선 로컬 단독 비교의 ONNX 파일만으로 승인 bundle이 준비됐다고 간주하지 않는다.

기본 관절 필터는 Human-Art manifest 값(비교 실험에서는 0.35), current-X는
0.30이다. 숫자를 VLM 수정 분기 기준이나 모델 간 동일 확률로 해석하지 않는다.
`--rtm-threshold`를 명시하면 해당 실험의 필터를 덮어쓸 수 있다.

```bash
export GEMINI_API_KEY='<본인 키를 비공개 환경변수로 설정>'
export HYBRID_GEMINI_MODEL='<사용 가능한 Gemini 모델 ID>'
python scripts/hybrid_pose_extract.py /path/to/rough.png \
  --output-dir out/hybrid-pose/example
```

실행하면 **원본 이미지 + RTM 표시 이미지 + 관절 JSON**이 Google Gemini에 전송되고
기본 단일 모드는 최대 1회 호출 비용이 발생할 수 있다. 모델을 명시해야 하며 실험에 사용한 ID는
`gemini-3.8-flash`였다. 계정에서 해당 모델을 사용할 수 있는지는 실행 전에 확인한다.
이 PR 준비/테스트에서는 실제 Gemini API를 호출하지 않았다.

기본값은 `--vlm-mode single --thinking-level low`다. `default`는 모델 기본값을
요청에 명시하는 것이 아니라 `thinkingConfig` 자체를 생략한다. 기본값이 High라고
가정하지 않는다. Low 지원 모델인지 호출자가 확인하며, 미지원 시 자동으로 바꾸지 않는다.

기존 출력 폴더는 덮어쓰지 않는다. 통신 시간초과는 서버에서 처리됐을 가능성이 있으므로
새 출력 폴더로 재실행하면 중복 호출/과금 가능성이 있다. 자동 재개나 무제한 재시작을 하지 않는다.

파일:

- `original.png`, `rtm-overlay.png`, `context.json`: 실제 전송 이미지/RTM 입력.
- `single_review_repair.response.json`, `single_review_repair.json`: 단일 호출 원문/파싱 응답. 로컬 검증 실패 시에도 보존한다.
- `evaluation.json`, `repair.json`: 같은 단일 응답에서 분리한 판단/패치. 별도 호출이 아니다.
- 기존 2단계 옵션에서만 `evaluation.response.json`, 필요한 경우 `repair.response.json`을 저장한다.
- `rtm.json`, `final.json`: 수정 전/후 관절. 오류 시 final 생성 안 함.
- `result.json`: 상태, 변경 이력, 검수 필요 여부, 단계별 시간, 모델과 프롬프트 해시.

이미지·원문·좌표는 민감할 수 있다. 기본 API 로그로 전송하지 않으며
출력은 gitignore된 `out/` 등 비공개 경로에 저장한다. 키는 파일/결과에 기록하지 않는다.

## 서버 코드에서 선택 실행

```python
from src.hybrid_pose.frontend import build_frontend
from src.hybrid_pose import HybridPosePipeline
from src.hybrid_pose.gemini import GeminiReviewer

# 키와 모델은 호출부의 비공개 설정으로 주입한다.
pose, threshold = build_frontend()  # humanart-m; 검증된 bundle/rollout 설정 필요
pipeline = HybridPosePipeline(pose, GeminiReviewer(api_key, model_id), threshold)
result = pipeline.extract(image_path)
if result["status"] == "ok":
    people = result["final"]["people"]
else:
    # 명시적 오류 처리. result["rtm"]은 미완료 제안일 뿐 최종 성공이 아니다.
    handle_error(result["error"])
```

HTTP 라우트/인증/큐/전체 요청 deadline/동시성 제한은 이 모듈에서 새로 만들지 않는다.
모델은 요청 전 초기화해 재사용할 수 있지만, ONNX 세션과 Gemini 사용량에 맞춘
운영 동시성·스케줄링은 호출부에서 제한해야 한다. CLI는 한 이미지씩 처리한다.

좌표는 **전체 이미지 x/y 각각 0..1000**, 왼쪽 위 원점. 해부학적 좌우와 COCO-17 순서.
`not_present`는 실제 부재와 근거 부족을 포함하므로 레코드 17개를 검출 17개로 세면 안 된다.
기존 `bbox`는 원래 RTM 제안의 관절 범위로 보존한다. 수정 후에는 이 bbox가
낡거나 면적 0일 수 있으므로 `result.keypoint_bounds`에서 최종 좌표의 범위를
별도로 제공한다. 이것도 인물 검출 박스가 아니며, 한 점만 있으면 면적 0/점이 없으면 null이다.
이 메타데이터 추가는 좌표 변경이나 시각 재평가가 아니다.

## 프롬프트와 실험 이식 범위

`src/hybrid_pose/prompts/`에는 기존 2단계 MD 세 개와 단일 호출 실험의
`공통_규칙_single.md`, `검토_single.md`, `수정_single.md`, `응답_계약_single.md`를 보존했다.
단일 호출에서는 이 네 문서를 실험과 같은 순서로 합쳐 실제 요청에 넣고 초기화 시
본문과 해시를 고정한다. 문서의 당시 실험 상태 문구도 입력 재현을 위해 보존했다.
기존 2단계 옵션에서는 공통+평가용 또는 공통+수정용을 사용한다.
GT, 데이터셋, 실험 응답, 개인 경로, macOS 잠자기 방지/자동화 코드는 포함하지 않는다.
원본 비율을 유지한 RGB PNG를 사용하고, 오버레이 폰트는 Linux에서도 사용할 수 있게 폴백한다.
NaN/Inf 점수는 null로 정리하고 스키마/ID/누락 처리 검사를 강화했다.

## 현재 확인한 결과와 한계

2026-10-07 로컬 실험에서 Set 1의 119/230/252 제외 **27컷**:

| 항목 | 값 |
| --- | ---: |
| RTM 유지 / 수정 경로 | 6 / 21컷 |
| RTM 평균 | 약 1.04초 |
| 전체 평균 / 중앙값 / p95 | 약 45.38 / 45.54 / 77.51초 |
| 20초 이내 | 1 / 27컷 |

**21컷 수정은 21컷 품질 개선을 뜻하지 않는다.** 사용자의 전반적 육안 반응은 좋았지만,
전체 컷에 대한 정형 품질 평가는 아직 없고 일부 손목/어깨 누락 및 RTM보다 나빠진 팔도 지적됐다.
초안/모델 유래 GT, bbox 기반 매칭 문제 때문에 정확도 개선율은 여기서 단정하지 않는다.
15°/5°는 과거 실험 비교용이며 서비스 반환 차단 기준이 아니다.

### 2026-10-08 단일 호출 Low 비교

Human-Art 동일 27컷에서 기본 추론 단일 호출은 27/27, Low는 24/27 성공했다.
다음 표는 양쪽 모두 성공한 **동일 24컷**만 비교한다(기본 추론은 설정 생략).

| 항목 | 단일 기본 추론 | 단일 Low |
| --- | ---: | ---: |
| 전체 평균 | 44.74초 | 14.82초 |
| 중앙값 | 37.36초 | 15.16초 |
| p95 (선형 보간) | 83.93초 | 18.88초 |
| 20초 이내 성공 | 2/24 | 23/24 |
| 수정 / 유지 | 16 / 8 | 23 / 1 |

각 실행 API는 27회다. 전체 시도 기준 Low의 20초 이내 성공은 **23/27**이며
094(Invalid removal), 030(Absent point must be null), 209(Invalid/conflicting edit)는
로컬 계약 검증에서 실패했다. 이를 자동 교정하거나 성공으로 바꾸지 않는다.
실패 포함 Low API 단계 누적은 193.49초다. 실행 시점이 다른 과거 결과 비교이고
전체 Low 품질 검수는 아직 없다. 수정 횟수 증가는 품질 개선을 뜻하지 않는다.
추론 토큰 필드는 Low 원문에서 제공되지 않아 추론량 0이라고 단정하지 않는다.

20초는 **모든 컷/운영 환경에서 달성한 보장이 아닌 목표**이다. 실험에는 호출 간격 대기가 포함되었고,
본 CLI에는 그 벤치마크 전용 15초 간격 대기를 넣지 않았다. 따라서 이식 후 속도가
위 값과 같거나 20초를 만족한다고 주장하지 않는다. 새 실측과 사용자 검수가 필요하다.
`timings_ms.total`은 warm 입력 읽기~최종 JSON 저장까지, RTM 초기화는
`rtm_initialization_ms`에 별도 기록한다. 클라이언트 업로드/다운로드, 요약 파일 쓰기는 제외.
실패한 단계의 소요 시간도 남긴다. `--timeout`은 단계별 네트워크 타임아웃이며
전체 20초 deadline이 아니다.

## 검증

```bash
python -m pytest tests/test_hybrid_pose_contracts.py tests/test_hybrid_pose_pipeline.py tests/test_hybrid_pose_frontend.py -q
```

API/모델 다운로드 없는 mock 검증으로 횟수 상한, 원본 보존, ID/좌표 검증,
누락 인물 추가, 부분 수정, 오류/시간초과, 불완전 응답, 입력 이미지·프롬프트 전송을 검사한다.
실제 모델 정확도나 운영 지연은 이 테스트로 입증되지 않는다.
