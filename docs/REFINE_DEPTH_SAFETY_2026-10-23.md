# 실제 타깃 깊이·관절·메시 보정 — 2026-10-23

직전 안전 FBX와 동일한 러프·후보·체형·손 조합에서, 실제 최종 FBX의 2D 관절
오차가 **추가 9.96% 감소**했다. 실제 Blender 적용, 몸·손 각도/깊이 검사,
전신 메시 검사, 최종 재수입과 receipt 발급까지 한 명령으로 완료했다.
배포·운영 등록은 하지 않았다. [구조화 실행 근거](../evaluation/refine_v3/depth_safety_applied_2026-10-23.json).

시각적 개선은 제한적이다. 새 Gemini 비교는 `preferred=uncertain`,
`depth_expressiveness_change=same`이라고 응답했다. **Z 정확도가 개선됐다는
검증 결과는 아니다.** 허리 측면 손 위치, 상체/시선 방향, 발의 지지·디딤 각도는
아직 러프와 차이가 있다.

## 자가 평가와 구현

직전 구현은 source BVH의 2D 정합과 오른팔 하나의 관통 보정에 집중했다.
실제 타깃 체형의 깊이와 팔·다리 관절의 변화를 최종 FBX에서 함께 제한하는
단계가 부족했다. 다음을 로컬 오프라인 적용 경로에 추가했다.

- `scripts/refine_target_depth.py`: 실제 Blender 뼈 길이와 고정 검색 카메라를
  사용한 타깃 정합. 어깨·고관절은 뼈 방향에 수직인 두 회전축, 팔꿈치·무릎은
  현재 굽힘 평면의 hinge를 사용한다. 2D로 알 수 없는 자유 축 비틀림을 변수로
  추가하지 않는다. 다리를 보정할 때 발 방향은 counter-rotation으로 유지한다.
- `converter/pose_safety.py`: 전 관절의 상대 회전·뼈 오프셋·stretch, 굽힘 방향,
  과도한 접힘, 카메라 깊이 순서와 이동량을 검사한다. 팔 관절 12°, 고관절·무릎
  8°, 발목 12°, 팔/다리 축 비틀림 4°의 보정 상한을 사용한다. 척추·목·머리와
  몸 보정 중 손·손가락은 잠근다. 손가락 적용은 기존 hand library의 curl/spread
  상한과 실제 Blender bake/reimport로 별도 검증한다.
- 깊이는 단일 러프에서 직접 관측했다고 만들지 않는다. 같은 후보의 직전 안전
  자세를 prior로 사용한다. 카메라 Z는 **멀어지는 방향이 양수**이며 Blender의
  높이 Z와 다르다. 충분한 기존 깊이 차이(몸통 길이 0.03 초과)가 있는 쌍은
  부호와 기존 차이의 최소 25%를 보존한다. 관절 깊이 이동은 몸통 길이의
  0.15 이내다. 이 기준은 선택한 3D prior를 보호하며 실제 Z 정답을 인증하지 않는다.
- 전체 메시에서 탈락한 triangle pair의 실제 skin region을 사용해 충돌한
  손/팔 사슬을 우선 줄이는 국소 backtracking을 추가했다. 전신 validator를
  다시 통과해야 하며, 어떤 충돌 쌍도 제외하거나 면을 축소하지 않는다.
- `converter/realize_pose_plan.py`: Blender가 실제로 표현하고 직렬화하는 rig를
  계획 입력으로 고정한다. near-pi 관절 roll의 FBX 수치 오차를 다루기 위해
  최대 3번의 제한된 roundtrip을 사용하며 기존 1e-5 행렬 기준을 유지한다.
- `scripts/plan_realized_pose_skin.py`: 실제 rig frame으로 독립 LBS maps와
  원본 rest의 세 corner director를 다시 계산해, 2mm 이내의 rest 보정을 계획한다.
- 최종 QA는 수정된 rest에서 Blender LBS를 한 번 평가하고, 고정된 rig에 새
  정점을 넣어 내보낸다. 실패한 계획 파일의 메시 정점은 채택하지 않는다.
  실제 뼈 행렬, 전체 메시, 독립 정점 재현, 최종 바이트를 모두 다시 검사한다.

각도 상한은 현재 source BVH를 Converter에 적용한 몸을 기준으로 한 보정 범위다.
모든 인체의 절대 가동범위나 연속 애니메이션을 인증하는 기준은 아니다.
v3 regional의 직접 관측/좌우 권한을 승격하지 않았으며 기존 full-body 계약을 쓴다.

## 실제 전후 측정

같은 저장 분석·후보·타깃, 같은 고정 카메라, 같은 이전 FBX의 몸통/골반 정규화를
두 파일에 적용했다. 출력된 FBX를 다시 읽어 추출한 실제 관절을 측정했다.

| 측정 | 직전 안전 FBX | 새 최종 FBX |
|---|---:|---:|
| COCO 몸 관절 12개 평균 정규화 2D 오차 | 0.1190934256 | 0.1072294404 |
| 양 손목 평균 오차 | 0.1894643448 | 0.1524926710 |
| 양 발목 평균 오차 | 0.1869318799 | 0.1736318145 |

정규화 2D 거리이며 일치율 백분율이나 독립 정답 평가가 아니다. 최적화 중 예측한
수치를 최종 파일 수치로 대신하지 않았다. 최종 깊이 제약은 12쌍 중 10쌍에
적용됐고 10/10 보존됐다. 나머지 2쌍을 확정 관측으로 승격하지 않았다.
최대 카메라 깊이 이동은 몸통 길이의 0.038722다.

새 Gemini `gemini-3.8-flash` 요청 1회에 원본, 이전/새 정면, 이전/새 측면,
새 사선 렌더를 보냈다(18.91초, 7,700 total tokens). 차이가 시각적으로 크지
않다고 평가했으며 신뢰도 medium이었다. 렌더 카메라와 러프의 시점 차이도
명시했다. 이를 안전 인증이나 Z 정확도 증명으로 사용하지 않았다.

## 안전 결과와 탈락 기록

| 실제 단계 | 메시 결과 | 최대 독립 정점 재현 오차 | 메시 검사 시간 |
|---|---|---:|---:|
| neutral-repair | mesh_validated | 0.000443mm | 2.20초 |
| body | mesh_validated | 0.001595mm | 2.31초 |
| hands / final | mesh_validated | 0.002173mm | 2.33초 |

- 기존 정점 허용치 **0.01mm**, 원본 rest 법선/면적, 접힘·교차·내부 포함 검사,
  3초 메시 검사 예산을 유지했다. 위 시간은 전체 적용 지연이 아니다.
- 52 bones, 24,994 vertices, 50,000 triangles. 최종 위반/미검증 사유 0건.
- 최종 body와 hands 행렬의 계획 대비 최대 원소 오차는 각각 **0 / 0**이다.
  1e-5 행렬 허용치를 늘리지 않았다. 최대 축 비틀림 변화는 약 0.061°다.
- 추가 rest 보정은 최대 **1.148899mm**, 106개 정점이다. 원본 릴리즈 FBX와
  원본 rest 기준은 보존하고 새 파생 스킨으로 식별한다.
- 큰 균일 보정은 오른손/오른쪽 허벅지의 실제 교차로 탈락했다. 해당 팔만 줄인
  국소 피드백안은 계획 메시를 통과하고 예측 오차 0.097818을 얻었으나,
  rig의 실제 직렬화가 1e-5 행렬 기준을 만족하지 못해 채택하지 않았다.
- 한 명령 실행은 더 작은 균일 0.25 step으로 다시 계획·실현·검증했다.
  실패 로그와 새 계획을 별도 폴더에 보존하고, 모든 최종 검사 후에만
  `applied.json`에 `applied=true, depth_safe=true`를 기록했다.

최종 파일 SHA256:
`d0476ba2306700b5e66865682a05aefba0720781d28ccfbf7b935882ee810099`

Receipt SHA256:
`65327fa994b3ea1f7a430b62da487de41d02821ca802aa584f1c6da7f9d7388d`

Source BVH SHA256은 직전과 동일하다:
`835b3eb6ab103385ce8b9ee1627014df950ec36671e20108f80e63008f3c0d17`

## 재현과 회귀

[직전 실제 적용 명령](REFINE_ACTUAL_ROUGH_APPLIED_2026-10-23.md#재현과-회귀)에
`--depth-safe`를 추가한다. 실제 실행은 기존 분석 JSON을 재사용하므로 provider
호출 0회다. 위 Gemini 시각 비교 1회는 별도로 실행하고 원본·렌더 SHA와 응답을
근거 JSON에 저장했다. 실제 데이터/모델/Blender와 새 출력 디렉터리가 필요하다.

```sh
OPENBLAS_NUM_THREADS=1 python scripts/apply_refine_rough.py \
  --analysis <saved-live-analysis.json> --data-root <matching-data-root> \
  --character <female-base-v0-3-4.fbx> --character-id female-base-v0-3-4 \
  --shape-registry <shape-bakes.local.json> \
  --hand-left support --hand-right relaxed --blender <Blender-executable> \
  --depth-safe --output-dir <new-output-directory>
```

현재 CLI의 depth-safe 경로는 derived skin planning input이 있는 단독 전신
인물을 대상으로 한다. 해당 전제나 어떤 안전 검사가 충족되지 않으면 중단한다.
다른 체형/러프/손 전체의 성공이나 제품 자동 적용을 주장하지 않는다.

- Converter **480 passed, 1 skipped**. Blender 전용 3개 모듈은 CPython 수집에서
  제외했으며 실제 Blender end-to-end는 위 실행으로 별도 검증했다.
- 추가 회귀: 보호된 척추/목/머리/손 변화, 팔/다리/발목 과회전, 축 비틀림,
  2D가 같아도 깊이가 뒤집힌 자세, 뼈 늘림, NaN 깊이 기준, 잘못된 hierarchy 차단.
- frozen Converter manifest **13개 전부 불변**, compileall/diff-check 통과.
- 기존 추론 경로는 이번 후속에서 변경하지 않았다. 직전 추론 2,501개 통과
  기록과 이번 Converter/실제 FBX 실행을 구분한다.
