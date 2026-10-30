# 리파인 v4 · 팔·다리 파츠 수정

[04J-318](https://linear.app/04jung/issue/04J-318). `refine-v4-parts.1`은 명시적으로
선택한 native 파츠를 적용하고 저장 FBX까지 검증하는 opt-in 경로다.
기존 `/analyze`, `/refine`, 검색 순위, 배포 설정은 바꾸지 않는다.

## 해결한 문제

이미 자세가 구워진 FBX에 다리 방향만 다시 적용하면 허벅지 회전과 피부 형태가
사라질 수 있다. 같은 canonical topology·가중치의 파츠는 골반 좌표에서 실제
피부와 이름이 있는 다리 rest frame을 함께 옮긴다. 다른 topology/가중치/길이는
거부한다. 현재 확정 파일과 과거 평가 baseline도 구분하여 실패 시 과거 초기
자세로 돌아가는 문제를 막았다.

* `current_asset.py`: 현재 확정 FBX의 SHA256·확인 출처 검증. 거부 시 이 파일의
  바이트를 그대로 반환한다. 사용자 형상 확인은 피부 검사 통과를 의미하지 않는다.
* `native_geometry.py`: canonical native 형상 전이, 반대 부위·상체 관절 보존,
  위아래 관계 검사, 고정 길이 발 전진 IK. 앞방향은 골반 +Z와 발끝 방향으로 확인한다.
* `leg_adapter.py`, `relations.py`: 인물 자신의 좌우, 교차 부위와 위쪽 다리,
  무릎 접힘을 구분한다. 방향 파츠의 무릎 교차와 발목 교차를 혼동하지 않는다.
* `pipeline.py`: 다리 → 팔 순서, 손가락까지 포함한 원자적 pose rollback,
  뼈 길이·선택하지 않은 관절·최종 피부/손 접촉 검사.
* `native_export.py`: 바뀐 체인만 rest bake. 재수입한 관절과 실제 피부 정점을
  각각 0.02mm 이내로 비교한다. 실패하면 현재 확정 파일을 반환한다.
* `src/arm_refine`: 팔 담당 브랜치의 검증 경계 `d09da4a`에서 가져왔다.
  접촉·손목·손가락과 부위별 자연어 계획을 포함한다. 이 커밋 이후 손 모양에 대한
  추가 피드백은 별도 진행 중이며 이 PR의 최종 승인 결과로 주장하지 않는다.

## 실행

Blender 5.2에서 검증했다. 요청 목록의 파일 경로는 JSON 파일 기준이다.
인체 FBX와 러프 원본은 저장소 정책에 따라 커밋하지 않는다.

```bash
blender --background --python-exit-code 1 \
  --python scripts/refine_v4_parts.py -- requests.json new-output
```

```json
[
  {
    "id": "left-foot-forward",
    "source_fbx": "historical-before.fbx",
    "source_sha256": "<historical SHA256>",
    "current_asset": {
      "fbx": "confirmed-cross.fbx",
      "sha256": "<confirmed SHA256>",
      "confirmation": "user_confirmed",
      "evidence": "사용자가 확인한 교차 자세"
    },
    "method": "advance_foot",
    "foot": {"side": "left", "distance_m": 0.05}
  }
]
```

`method`는 `retain_current`, `native_geometry`, `advance_foot`, `directions`,
`arms`를 지원한다. `retain_current`는 기존 파일을 유지하는 것이며 새 자동 수정
성공으로 집계하지 않는다. `native_geometry`의 `part`에는 donor FBX/SHA256,
`sides`, `intent.source`(`user_confirmed` 또는 `reviewed_library`),
`intent.reference`를 넣는다. 교차라면 `intent.sentence`, `top_side`,
`cross_region=knee`와 실제 무릎 관계가 일치해야 한다. 검증하지 못하는 발목·종아리
교차는 자동 재구성하지 않고 현재 파일을 유지한다.

`directions`는 `legs`에 `sentence`, `selection`, `source`, `reference`,
`part_file`을 받는다. 팔 단계는 `arms`에 좌우별 문장과 `rig_profile` JSON을
받는다. 현재 통합 runner의 팔 조합은 우산/휴대폰, 무릎 위 손/내린 손이다.
그 외 `language_plan.plan_language_arms`와 `language_adapter.apply_language_plan`은
별도 명시 선택 API로 제공하며 현재 runner에서 자동 호출하지 않는다.

`after.fbx`, `receipt.json`, `summary.json`에 실제 적용 여부와 거부 이유를 남긴다.
`release_allowed=false`는 검수용 경로라는 뜻이다. 점수로 기하 검사를 대체하지 않는다.
`author_cross_leg_preview.py`는 추가 사용자 지시로 만드는 별도 **형상 후보**이며
피부 검사를 우회한 자동 적용 결과가 아니다.

## 실험·점수

[2026-10-29 실험 결과](../experiments/refine_v4/2026-10-29/README.md)에 실제
저장본 해시, 전후 4시점 이미지, 고정 채점 코드·배점·원판정과 부분 재평가를 넣었다.

다리 7건 중 같은 목표로 비교 가능한 6건의 평균은 **65.00–73.33 →
76.67–86.67 /100**이다. 같은 비대상 부위의 불확실성을 전후 함께 유지한
순증 범위는 평균 **+10–15점**이다. AI 보조 부분 판정이며 사람의 점수 검수 전이다.
T022는 사진 참고와 후속 사용자 지시로 목표가 바뀌어 원래 러프와 직접 비교하지 않는다.

이 7건은 사용자 지시로 작성·복원한 형상 실험이다. 새 CLI에서는 별도로 T043
형상 교체, T050 왼발 전진, T002/T018 팔 적용의 **4건 저장·재불러오기**와
T041 거부 시 확정 파일 보존을 재현했다. 7건 전체를 새 자동 진단 성공으로
집계하지 않는다. 기존 자동 11건 재시험 결과도 별도 JSON에 보존했다.
상반신 /45와 전신 /100, 다른 팔 실험의 중복 컷을 합산하지 않는다.

## 검증·제약

* `python -m pytest tests/ --ignore=tests/converter -q`
* `python tests/test_smoke.py`, `python -m compileall src api scripts`
* Blender: `tests/converter/verify_refine_v4_native.py -- SOURCE.fbx NEW_OUTPUT PROFILE.json`
* Blender: `tests/converter/verify_refine_v4_geometry.py -- CURRENT.fbx DONOR.fbx NEW_RESULT.json`

피부 검사는 capped weighted region의 자동 triage다. 닫힘 실패는 거부한다.
오목한 절단면의 cap은 과대 판정할 수 있고 같은 손 내부 손가락끼리는 검사하지 않는다.
기존 작성 자산의 피부 경고는 실험 JSON에 공개했다. 접지력·체중 균형은 측정하지 않았다.
좌우·접촉·가림이 불분명한 자연어를 자동 정답으로 바꾸지 않는다.
