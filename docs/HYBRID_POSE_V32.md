# Hybrid Pose v3.2 — RTM 유지 + 선택적 VLM 보완

이 PR은 실험에서 검수한 파이프라인을 **독립 실행 가능한 모듈/CLI**로 이식한다.
기존 `/analyze`, 3D 포즈 검색·리타게팅·배포 설정은 바꾸지 않는다.
운영 API에 연결하거나 기본 경로를 바꾸는 것은 별도 검증/PR 대상이다.

## 흐름

```text
원본 러프 → fresh RTMPose → Gemini 평가 1회
                              ├ pass → RTM 그대로
                              ├ review_needed → RTM + 검수 필요
                              └ repair → Gemini 부분 수정 1회 → 코드 검증
                                                  ↓
                                  최종 관절 JSON + 변경 이력 + 시간
```

- 평가 대상: 인물 누락, 찾을 수 있는 관절 누락, 명백한 위치/연결 오류, 가짜/중복 인물.
- 신뢰도가 낮다는 이유만으로 수정하지 않는다. 0.3은 RTM 좌표 유효성 필터이고 VLM 분기 기준이 아니다.
- 미세 각도는 수정 조건이 아니다. GT/정답 인물 수를 입력하지 않는다.
- 수정은 지적된 관절과 같은 팔/다리의 필수 연결점만 허용한다.
- 수정하지 않는 좌표·상태·RTM 점수와 기존 인물 ID는 유지한다. 수정점의 RTM 점수는 제거한다.
- 새 인물은 별도 ID를 부여하고 COCO-17 항목을 검증한다. 화면 밖/근거 부족 관절은 null이다.
- 수정 후 **VLM 재평가 없음**. 각 단계 API 최대 1회, 자동 재시도/모델 변경 없음.
- 오류는 `status=error, final=null`; 보존한 `rtm`을 성공 결과로 둔갑시키지 않는다.
- `status=ok`는 실행/형식 검사 완료일 뿐 사람의 품질 승인이나 각도 합격이 아니다.

## 실행

저장소 루트에서 기존 requirements를 설치한다. RTM은 기존
`src.pose.RTMPoseModel`(`rtmlib.Body(performance, onnxruntime, cpu)`)을 사용한다.
첫 모델 초기화에는 모델 다운로드가 필요할 수 있다. mock으로 자동 대체하지 않는다.

```bash
export GEMINI_API_KEY='<본인 키를 비공개 환경변수로 설정>'
export HYBRID_GEMINI_MODEL='<사용 가능한 Gemini 모델 ID>'
python scripts/hybrid_pose_extract.py /path/to/rough.png \
  --output-dir out/hybrid-pose/example
```

실행하면 **원본 이미지 + RTM 표시 이미지 + 관절 JSON**이 Google Gemini에 전송되고
1~2회 호출 비용이 발생할 수 있다. 모델을 명시해야 하며 실험에 사용한 ID는
`gemini-3.8-flash`였다. 계정에서 해당 모델을 사용할 수 있는지는 실행 전에 확인한다.
이 PR 준비/테스트에서는 실제 Gemini API를 호출하지 않았다.

기존 출력 폴더는 덮어쓰지 않는다. 통신 시간초과는 서버에서 처리됐을 가능성이 있으므로
새 출력 폴더로 재실행하면 중복 호출/과금 가능성이 있다. 자동 재개나 무제한 재시작을 하지 않는다.

파일:

- `original.png`, `rtm-overlay.png`, `context.json`: 실제 전송 이미지/RTM 입력.
- `evaluation.response.json`, `repair.response.json`: 반환된 원문; 수정 생략 시 후자는 없음.
- `evaluation.json`, `repair.json`: 모델의 판단과 수정 제안.
- `rtm.json`, `final.json`: 수정 전/후 관절. 오류 시 final 생성 안 함.
- `result.json`: 상태, 변경 이력, 검수 필요 여부, 단계별 시간, 모델과 프롬프트 해시.

이미지·원문·좌표는 민감할 수 있다. 기본 API 로그로 전송하지 않으며
출력은 gitignore된 `out/` 등 비공개 경로에 저장한다. 키는 파일/결과에 기록하지 않는다.

## 서버 코드에서 선택 실행

```python
from src.pose import RTMPoseModel
from src.hybrid_pose import HybridPosePipeline
from src.hybrid_pose.gemini import GeminiReviewer

# 키와 모델은 호출부의 비공개 설정으로 주입한다.
pipeline = HybridPosePipeline(RTMPoseModel(), GeminiReviewer(api_key, model_id))
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

`src/hybrid_pose/prompts/`에는 실제 v3.2 실험 입력 MD 세 개를 보존했다.
문서의 “초안/실행기 미연결” 문구는 당시 스냅샷이며 현재 모듈은 해당 본문을 실제 호출에 넣는다.
공통+평가용 또는 공통+수정용 본문을 초기화 시 고정하고 해시를 기록한다.
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

20초는 **미달성 목표**이다. 위 실험에는 호출 간격 대기가 포함되었고,
본 CLI에는 그 벤치마크 전용 15초 간격 대기를 넣지 않았다. 따라서 이식 후 속도가
위 값과 같거나 20초를 만족한다고 주장하지 않는다. 새 실측과 사용자 검수가 필요하다.
`timings_ms.total`은 warm 입력 읽기~최종 JSON 저장까지, RTM 초기화는
`rtm_initialization_ms`에 별도 기록한다. 클라이언트 업로드/다운로드, 요약 파일 쓰기는 제외.
실패한 단계의 소요 시간도 남긴다. `--timeout`은 단계별 네트워크 타임아웃이며
전체 20초 deadline이 아니다.

## 검증

```bash
python -m pytest tests/test_hybrid_pose_contracts.py tests/test_hybrid_pose_pipeline.py -q
```

API/모델 다운로드 없는 mock 검증으로 횟수 상한, 원본 보존, ID/좌표 검증,
누락 인물 추가, 부분 수정, 오류/시간초과, 불완전 응답, 입력 이미지·프롬프트 전송을 검사한다.
실제 모델 정확도나 운영 지연은 이 테스트로 입증되지 않는다.
