# 러프 데이터 선순환 — 라이브러리 공백 찾기

작가가 올린 러프로 검색해도 맞는 3D 포즈가 없던 사례를 모아, 여러 사용자에게서
되풀이되는 자세를 라이브러리에 더하고, 더한 뒤 실제로 메워졌는지 다시 잰다.
이 문서는 그중 **공백을 찾고 재는 부분**(`pose_gaps/`)과, 그 데이터가 지켜야 할 규칙을 다룬다.
후보 제작·검수는 `pose_curation`(직접 제작 포즈와 같은 검수 기준), 번들·배포는
`docs/POSE_LIBRARY_BUNDLE.md`를 따른다.

```
[운영] /analyze → BFF RDS analysis_people(관절·점수·태그·거리·반응) — 1년, 삭제·철회 시 함께 삭제
   │  GET /v1/admin/gaps/observations  (비식별·export마다 다른 가명·감사 기록)
   ▼
[로컬] python -m pose_gaps pull → analyze → report
   적격 판정 → 재검색 충실도 → 공백 라벨 → 좌우 반전 불변 군집 → 설치 3곳 이상만 ID 없는 집계
   ▼
[제작·검수] pose_curation: 라이선스 풀 발굴·구도 맞춤·프리셋 탐색 → render → qa → AI 1차·사람 서명
   ▼
[배포] build_pose_bundle.py → pose_gaps privacy-scan → pose_gaps gate → record-gate → deploy
   ▼
[측정] python -m pose_gaps measure → 군집별 메움률을 집계 이력에
```

## 데이터 규칙 (먼저 읽을 것)

- **사용자 원본 이미지를 로컬에 복사하지 않는다.** BetaData 버킷은 90일 만료·동의 철회 삭제를
  약속한 저장소다. 로컬 사본은 그 약속 밖에 놓인다. 공백 분석에 필요한 것은 운영이 이미
  저장한 파생 데이터(관절·점수·태그·검색 거리)뿐이고, 이것만 export로 받는다.
- **사용자에게서 나온 값으로 포즈 ID·파일 이름·설정을 만들지 않는다.** `user_<sha12>`처럼
  입력 해시 앞자리를 쓰면 `jobs.input_sha256`으로 작업까지 다시 연결된다.
- **러프 관절을 3D 목표로 쓰지 않는다.** 관측은 공백을 찾고 후보의 적합도를 재는 데만 쓴다.
  3D 포즈는 라이선스 모션·직접 제작 프리셋에서만 나온다.
- **개별 관측은 로컬에 오래 두지 않는다.** 스냅샷은 최신 하나만 남기고 14일이 지나면 지운다.
  분석은 7일 넘은 스냅샷으로는 돌지 않는다. 운영에서 지워진 데이터가 로컬에서 사라지기까지
  최대 14일이다. 정리 PC 디스크는 암호화돼 있어야 한다.
- **오래 남기는 것은 ID 없는 집계뿐이다.** 서로 다른 설치 3곳 이상에서 모인 군집의 평균 자세,
  건수, 태그 분포. 설치 3곳 미만이 본 관절은 평균에서 비우고, 3곳 미만인 태그 값은 `_other`로 합친다.

확인: `python -m pose_gaps privacy-scan data/curation config --strict`
(`--hash-list`로 사용자 입력 SHA-256 목록을 주면 그 값과 앞 12자도 찾는다. 목록은 메모리에만 올린다.)

## 명령

```bash
export STANDIN_ADMIN_TOKEN=…            # 디스크에 쓰지 않는다
python -m pose_gaps pull --base-url https://api.standinpose.com --days 90
python -m pose_gaps analyze --production data/bundles/current \
  --curated data/curation/library/poses.db           # 정리 라이브러리: 배포 대기 포즈 확인(선택)
# → data/gaps/snapshots/<시각>/report.html, data/gaps/aggregates/clusters.json

python -m pose_gaps gate --baseline data/bundles/current --candidate data/bundles/next \
  --coverage-extraction data/curation/coverage/<날짜>/extraction --out data/gaps/gate/next.json
python scripts/build_pose_bundle.py record-gate --bundle data/bundles/next --report data/gaps/gate/next.json

python -m pose_gaps measure --production data/bundles/next --parent data/bundles/current
```

데이터 폴더는 `--root` > `POSE_GAPS_DATA_DIR` > `data/gaps`. 임계값은 `config/pose_gaps.json`
(버전이 보고서·집계에 남는다).

## 분석 단계

1. **적격 판정**(`eligibility.py`): VLM 슬롯, 전체 이미지 추출, state valid/partial, coverage
   full/reduced, 전신 검색, 얽힘 아님, 관절이 틀렸다는 피드백(`skeleton_wrong`·`person_missing`)
   없음, 그리고 `pose_curation/coverage`와 같은 정량 조건(몸통 4점, 몸 관절 10/12, 몸통 20px).
2. **재검색 충실도**(`replay.py`): 운영과 같은 쿼리 피처
   (`normalize_skeleton(관절, raw 점수, 0.3, evidence 마스크)`)와 운영이 실제로 쓴
   `search_mask`로 다시 검색한다. 같은 라이브러리 버전의 관측은 Top-1 pose·거리가 export와
   99% 이상 같아야 한다. 아니면 멈춘다 — 그 상태의 공백 목록은 믿을 수 없다.
3. **라벨**: `not_gap`(≤ τ_soft 0.25), `extraction_suspect`(≥ 0.6, "후보가 엉뚱함" 피드백 없음),
   `filled_pending_deploy`(정리 라이브러리에서는 ≤ τ_fill 0.20), `gap_open`(나머지, 0.35 초과면 strong).
4. **군집**(`cluster.py`): coverage class별 complete-linkage(ε 0.22), 원본과 좌우 반전 중
   가까운 거리. 크기는 관측 수가 아니라 서로 다른 설치 수.
5. **집계**(`aggregates.py`): 설치 3곳 이상 군집만. 같은 자리의 군집은 다음 주기에도 키를
   유지하고 이력만 늘린다.

## 게이트 (`pose_gaps gate`)

포즈를 더하면 Top-1 거리는 줄 수밖에 없어서, 공백 메움률이 아니라 **기존 결과를 망치지 않는지**로
판정한다. 쿼리는 최근 스냅샷의 적격 관측 전체와 제공 러프 coverage 세트(`user_*` 파일 제외).

| 검사 | 통과 조건 |
|---|---|
| 나빠진 쿼리 | Δ > +0.02인 쿼리 ≤ 2% |
| high 이탈 | 기준 ≤ 0.25였던 쿼리가 0.25를 넘지 않음(의도적으로 뺀 포즈만 +0.05까지 허용) |
| 쏠림 | 원래 잘 맞던 쿼리 안에서 어떤 family도 Top-1 점유율 > max(5%, 기준×3) 아님 |
| 반전 짝 | 신규 포즈마다 `_mirror` 짝 또는 같은 family의 짝(`--waive-mirror`로 면제) |
| 자산 | 후보 번들에 모든 포즈의 BVH·썸네일 4장 |
| 결정성 | 두 번 계산해도 같은 결과 |

공백 메움률(`gap_filled_before/after`)은 보고만 한다. 보고서에는 집계만 남는다.

## BFF export 계약 (schema 1)

`GET /v1/admin/gaps/observations?days=<1..365>&cursor=<opaque>` — `X-Beta-Admin-Token`, 허용 검수자만,
페이지마다 `admin_access_audit` 기록. 응답:

```json
{"schemaVersion": 1, "exportId": "…", "window": {"days": 90}, "generatedAt": "…",
 "retention": {"localTtlDays": 14, "maxSnapshotAgeDays": 7},
 "items": [ … ], "nextCursor": "…" | null}
```

항목(`pose_gaps/observations.py`가 읽는 형식):

```json
{"obs": "o_<22>", "inst": "i_<22>", "observed_on": "2026-09-30", "expires_on": "2027-09-30",
 "versions": {"pose_library": "lib-…|v1", "feature": 1, "pose_model": "…", "vlm_model": "…",
              "vlm_prompt": "…|null", "deployment": "…"},
 "cut": {"route": "core", "count_confidence": "high", "person_count": 2,
         "tags": {"shot": "…", "action": "…", "view": "…", "relationship": "…"}},
 "person": {"keypoints": [[x, y] × 17], "raw_scores": [17], "effective_scores": [17],
            "evidence_mask": [17], "search_mask": [17],
            "coverage_class": "full", "skeleton_state": "valid", "skeleton_source": "full_image",
            "slot_origin": "vlm", "lower_body_observed": true, "confidence": "low",
            "fallback_mode": "soft", "search_scope": "full_body", "distance_metric": "pos",
            "search_stability": "not_required", "rank_distance": 0.41, "confidence_threshold": 0.45,
            "quality_reasons": [], "tags": {"action": "sitting", "view": "side", "source": "vlm_person"},
            "scope": {"detected": "full", "source": "vlm_person"}},
 "candidates": [{"pose_id": "…", "view": "side", "rank": 1, "distance": 0.41, "match_level": "low"}],
 "behavior": {"selected_rank": null, "exported": false, "selected_refined": null,
              "job_feedback": "candidates_irrelevant"}}
```

- 가명 `obs`·`inst`는 export마다 새 salt로 만든 HMAC이다. 같은 export 안에서만 같은 값이라,
  설치 수는 셀 수 있지만 export끼리나 운영 ID와는 이을 수 없다.
- **넣지 않는 것**: job_id, installation_id, input_sha256, S3 key, bbox, 이미지 크기. 날짜는 일 단위,
  관절은 0.1px 반올림. 받는 쪽도 금지 키와 원본 ID 모양 문자열이 하나라도 있으면 스냅샷을 만들지 않는다.
- `evidence_mask`·`search_mask`는 운영 `quality_trace.evidence_valid_joint_mask`·
  `search_valid_joint_mask`, `raw_scores`는 `analysis_people.raw_scores_json`이다.
  `skeleton_json.scores`는 refine이 막히면 0으로 저장되므로 쿼리 재현에 쓰지 않는다.
- 행은 BFF의 원본 테이블 위 뷰에서 나오므로, 작업 삭제·동의 철회·365일 정리와 함께 사라진다.

## 인물별 태그와 프롬프트 전환

공백 군집의 태그 히스토그램(`person_action`·`person_view`)은 export 항목의 `person.tags`에서 온다.
값은 추론 `PersonOut.person_tags`이고 `source`가 출처를 알려 준다(`docs/API_CONTRACT.md`).

- `p1-scope`(지금 운영): 인물별로 묻지 않는다. 사람이 1명인 컷만 VLM이 실제로 말한 컷 값을
  `legacy_cut`으로 쓰고, 여러 명인 컷은 `unknown`이다.
- `p2-person-tags`: 인물마다 `vlm_person`으로 받는다. 켜기 전에 아래 게이트를 통과해야 한다.

```bash
# 평가 데이터셋(selected12와, provider에 보낼 권리가 확인된 제공 러프만)에서 A/B를 번갈아 5회씩
python -m standin_eval run vlm-compare --dataset <id> --provider gemini --a p1-scope --b p2-person-tags
```

같은 프롬프트끼리의 차이(A↔A)를 잡음 바닥으로 두고 A↔B를 잰다(`standin_eval/vlm_compare.py::GATE`).

| 항목 | 통과 기준 |
|---|---|
| B 파싱 실패 | 0건 |
| route 변화 | 잡음 바닥 + 2%p 이하 |
| 인원수 일치율 | 잡음 바닥 − 3%p 이상 |
| 박스 IoU 중앙값 | 하락 0.03 이하 |
| 얽힘 판정 변화 | 잡음 바닥 + 1건 이하 |
| 지연 p95 | +15% 이하 |
| 인물 배열 정렬률 | 95% 이상 |

통과하면 staging부터 GitHub environment 변수 `VLM_PROMPT_VERSION=p2-person-tags`를 두고 배포
워크플로를 다시 실행한다(`.github/workflows/deploy.yml`). beta도 같은 방식이고, 7일 동안 지표를
본다. 되돌릴 때는 변수를 지우고(기본 `p1-scope`) 워크플로를 다시 실행한다. 응답의
`inference_metadata.vlm_prompt_version`과 BFF에 저장된 값으로 어느 프롬프트가 답했는지 구분된다.
