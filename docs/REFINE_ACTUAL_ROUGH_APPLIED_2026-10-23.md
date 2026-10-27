# 실제 러프 적용과 최종 FBX 안전 검증 — 2026-10-23

후속: [깊이·관절·메시 안전 보정](REFINE_DEPTH_SAFETY_2026-10-23.md)에서 같은 후보의
실제 최종 FBX 오차가 추가 9.96% 감소했다. 새 Gemini 검수의 시각적 개선 판단은
불확실하므로 수치 개선과 구분해 기록했다.

`124629.png`의 실제 Gemini/Human-Art 분석으로 포즈를 적용한 **female BASE 파생
스킨 + 왼손 support / 오른손 relaxed 1개 조합**이 최종 FBX 전신 검증을 통과했다.
중립 포즈로 대체하지 않았다. 같은 입력을 한 명령으로 다시 실행해 별도의 최종
파일과 검증 receipt도 얻었다. 배포, 운영 프로필 등록, 운영 플래그 변경은 하지 않았다.

이 결과는 [전날 실패 보고서](REFINE_V3_LIVE_GEMINI_ROUGH_2026-10-22.md)의 후속이다.
당시 실제 러프 안전 승인 0건이라는 기록을 소급해서 바꾸지 않는다.
원시 수치와 검증 기록은 [실행 근거 JSON](../evaluation/refine_v3/actual_rough_applied_2026-10-23.json)에 있다.

## 원인과 수정

1. 기존 limb observability는 작은 회전에 따른 목표 방향 loss 변화를 사용했다.
   목표에 이미 가까운 경우나 반대 방향인 경우 미분이 사라지고, 여러 뼈의 변화가
   상쇄되어 실제로 움직일 수 있는 팔을 비관측으로 판단했다. 이를 ±5° probe의
   투영 관절 이동량 / 3D 이동량으로 바꿨다. 원근 카메라는 baseline 깊이를 사용해
   같은 물리 단위로 비교한다. 기존 관측 임계값, 회전 한계와 안전 기준은 유지했다.
2. 기존 1순위는 양팔을 굽힌 자세였다. 저장된 검색 결과 상위 5개 중 기존 정책상
   허용된 4개를 실제 `/refine` 처리기로 보정·비교했다. 세트 포즈 1개는 제외했다.
   이 명시적인 오프라인 비교에서 기존 2순위 `rokoko_FootTapping_mixamo_00040`을
   선택했다. 제품 `/refine`에서 사용자가 선택한 포즈를 몰래 교체하지 않는다.
3. 실제 체형에 굽힐 때 팔·어깨의 급격한 스킨 가중치 변화가 접힘을 만들었다.
   원본 릴리즈는 보존하고 팔 주변 7,116개 정점의 가중치를 국소 확산한 파생 스킨을
   만들었다. 최대 가중치 변화는 0.42234다. 기존 발끝 6개 정점 보정에 더해
   106개 정점을 최대 0.9172mm 보정했다. 추가 rest 보정 상한은 2mm다.
4. 손과 허벅지 관통은 오른팔을 6° 벌려 해소했다. 0°와 3° 시도는 충돌로 탈락했다.
   전체 목표 몸 포즈를 derived rest에서 한 번 굽고 손을 별도 적용한다. 이미 구운
   메시를 다시 몸 변형에 사용하는 프로토타입은 독립 재현 검증에서 탈락했고 수정했다.

계획 단계의 optimizer 성공만으로 파일을 승인하지 않는다. 실제 Blender 적용,
각 단계의 독립 LBS 재현, 전체 메시 검사, 최종 FBX 재수입 및 바이트 receipt를
모두 통과해야 `applied.json`에 `applied=true`를 기록한다.

## 실제 최종 파일의 정합

| 측정 | 이전 실제 FBX | 수정 실제 FBX |
|---|---:|---:|
| 몸 관절 12개 평균 정규화 2D 오차 | 0.1310005 | 0.1190934 |
| 양 손목 평균 정규화 2D 오차 | 0.2177821 | 0.1894643 |

최종 FBX의 몸 관절 오차는 **9.1% 감소**했다. COCO 관절 5–16의 평균 거리를
투영 몸통 길이로 나눈 값이다. 각 후보의 기존 검색 카메라를 고정하고 비교했다.
관측은 저장된 실제 모델 분석이며 독립 정답이나 정확도 백분율이 아니다.
이전 후보는 yaw 0° orthographic, 새 후보는 yaw 45° perspective / distance 8 torso다.
렌더의 anatomical 정면과 이 수치 평가 카메라는 구분한다.

변환 전 BVH 비교에서는 0.1485050 → 0.0742804로 약 50% 감소했다. 이는 최종 FBX의
개선율이 아니다. 후보 변경, solver 수정, 타깃 보정이 함께 적용된 결과이므로
9.1% 전체 개선을 observability 수정 하나의 인과 효과로 주장하지 않는다.

실제 Gemini `gemini-3.8-flash`에 원본 러프, 이전 렌더, 수정 정면·사선 렌더를
새로 보내 1회 비교했다(19.02초, total 5,616 tokens). 수정 결과를 선호했고
한쪽 팔이 내려간 배치를 개선점으로 평가했다. 허리 측면에 있어야 할 손이 하복부
앞쪽에 있고, 발·다리 지지 각도가 다르며, 시점 비교에 제약이 있다고 지적했다.
손모양 confidence는 low다. VLM 평가는 안전 인증이나 직접 관측 권한이 아니다.

## 엄격 검증 결과

| 단계 | 전체 메시 | 독립 정점 재현 최대 오차 | 메시 검사 시간 |
|---|---|---:|---:|
| neutral-repair | mesh_validated | 0.000443mm | 2.20초 |
| body | mesh_validated | 0.003668mm | 2.33초 |
| hands / final | mesh_validated | 0.007329mm | 2.34초 |

기존 `full-body-posed-skin-v2`, 0.01mm 재현 허용치, 원본 rest의 세 corner
director·면적, 접힘·교차·내부 포함 검사와 3초 메시 검사 예산을 유지했다.
52 bones, 24,994 vertices, 50,000 triangles이며 최종 위반과 미검증 사유는 모두 0이다.
이 시간은 개별 메시 검사 시간이며 전체 적용 지연이 아니다.

- 원본 러프 SHA256: `ba92e47ea1600905a81b5973813f5c711087e479898dec0b230ae5716de3ccc6`
- source BVH: `835b3eb6ab103385ce8b9ee1627014df950ec36671e20108f80e63008f3c0d17`
- 제공·렌더·Gemini 검수한 final FBX: `310e8eaf166b21d824a5fc070e2d8dd222d82f55fdafa34ad8111af9319c0fa8`
- 해당 receipt: `f9616e4ee718fb431e5ee15956f52435e91a29c82f2d7002da1269bf98852ec3`
- 별도 한 명령 재현 final FBX: `90aa53dbff4e8f39178ff15650c786ea9a08619e46ea580d8f08b4b8191ca894`
- 재현 receipt: `316e71a279fb8e212dd5a72732c4ddf45042e756f5dbf00c68c119978eb4a9e1`

두 실행은 같은 조합이다. 성공 조합 2개로 세지 않으며, 각 실행의 프로필·최종
바이트 식별자를 구분한다. 승인 범위는 해당 최종 스냅샷이다. 다른 체형, 손,
러프나 연속 애니메이션을 포괄 인증하지 않는다. v3 regional의 관측 권한은
승격하지 않았으며 기존 full-body refine 계약을 사용했다.

## 재현과 회귀

저장된 실제 분석 JSON, 그 분석에 사용한 DB/BVH data root, 원본 체형과 registry,
Blender가 필요하다. Python에는 프로젝트 추론 의존성과 NumPy/SciPy가 필요하다.
출력 디렉터리는 새 경로여야 한다. 파일 경로는 해당 환경의 입력으로 지정한다.

```sh
OPENBLAS_NUM_THREADS=1 python scripts/apply_refine_rough.py \
  --analysis <saved-live-analysis.json> \
  --data-root <matching-data-root> \
  --character <female-base-v0-3-4.fbx> \
  --character-id female-base-v0-3-4 \
  --shape-registry <shape-bakes.local.json> \
  --hand-left support --hand-right relaxed \
  --blender <Blender-executable> \
  --output-dir <new-output-directory>
```

이 명령은 분석을 재사용하므로 provider 호출 0회다. 새 Gemini 시각 비교 1회는
별도의 실제 검수이며 위 실행 근거 JSON에 입력 이미지 SHA와 응답을 기록했다.
선택 후보, 각 단계 로그, 계획, 실패 기록, final QA, `applied.json`을 남긴다.
현재 대상은 기존 권한이 있는 단독 전신 인물이며 다른 입력의 성공을 보장하지 않는다.

- 추론: 2,501 passed, 10 skipped, 28 subtests passed.
- Converter: 463 passed, 1 skipped. Blender 전용 모듈 3개는 CPython 수집에서 제외했다.
- 실제 Blender end-to-end 적용을 별도로 두 번 완료했다.
- frozen Converter SHA256SUMS.v325 13개 불변.
- 추가 회귀는 관측의 목표 방향·뼈 길이·카메라 단위 독립성, 순수 깊이 이동 기권,
  계획의 입력 식별·변형 상한, 비활성 영역 보존, 실제 교차 차단을 검사한다.
