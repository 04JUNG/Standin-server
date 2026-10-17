# 실제 러프의 부위 윤곽·가림·공통 포즈 체형 비교 — 2026-10-17

실제 러프 28명에서 **세부 체형 확정 0명, 보류 28명**이다.
최신 체형 20종을 같은 BVH·시점에 놓고 비교하는 경로까지 구현했지만,
현재 성능으로 실제 러프의 세부 체형 인식이 해결됐다고 결론 내릴 수 없다.
기존 API/BFF와 독립된 명시적 오프라인 실험이며 배포·머지·런타임 활성화는 하지 않았다.

## 실행 결과

| 단계 | 인원/수치 | 의미 |
|---|---:|---|
| 실제 러프 입력 | 28명 | 기존 camera_real_rough 평가셋의 인물 슬롯 |
| 기존 소유권 조건 통과 | 18/28 | protocol owner_verified AND 기존 관찰 ownership_ambiguous=false |
| 노출/밀착 부위 판독 | 3/18 | 가시성 판독만 통과, 윤곽 검증과 다름 |
| 원본 선으로 검증한 윤곽 | 1/28 (3.57%) | 131040:p0의 몸통·팔·종아리 6부위 |
| 수치 비교 | 1/28 (3.57%) | 공통 포즈 2개 × 최신 20종 = 최종 40매 |
| 세부 체형 후보 확정 | 0/28 | 1·2위 차이/부위 제거 안정성/포즈 일관성 부족 |
| 자동 적용 | 0/28 | 실험 출력의 자동 적용은 항상 금지 |
| 보류 | 28/28 | 소유권 10, 윤곽 부족 17, 비교 후 근거 부족 1 |
| Top-1 / Top-3 정확도 | null / null | 작가의 허용 체형 정답 없음 |
| 분할 IoU | null | 정답 마스크 없음; SAM의 predicted_iou로 대체하지 않음 |

이전 14명 후보 보류/14명 근거 부족 실험과 게이트·관찰 단위가 다르므로
이번 28명 보류를 동일 기준의 정확도 회귀로 해석하지 않는다.
3.57%는 비교 도달률이며 정확도가 아니다. 개발 중 같은 입력을 재사용했다.
이번 데이터는 독립 검증셋이 아니고 임계값은 실제 러프 정답으로 보정되지 않았다.

## 구현과 판독 기준

- `rough_evidence.py`: 몸통, 좌우 상완/전완/허벅지/종아리 9부위의
  exposed/fitted/loose_clothing/occluded/out_of_frame/symbolic/uncertain 판독.
  Gemini에는 원본과 검출기 관절의 번호 가이드를 제공한다.
  관절 좌표, 나이, 성별, 프리셋, 보이지 않는 해부학은 생성하지 않는다.
  새 키를 이용한 가시성 요청 6배치 모두 성공했고 실제 모델은 `gemini-3.5-flash-lite`였다.
- `rough_segmentation.py`: 검출기 관절의 점/박스와 이웃 인물 음성 점을 사용한
  로컬 SAM 2.1 tiny 분할. 자기 신체의 다른 부위를 모두 음성 점으로 지정하면
  겹친 사지와 선화가 과도하게 제거되어 그 방식을 폐기했다.
  인물 전체 마스크는 검토용이며 체형 점수에 그대로 쓰지 않는다.
- 표면이 보이는 부위에서만 20%~80% 구간의 9개 수직 단면을 조사한다.
  모델 mask quality 0.70 이상이어도 원본 경계와의 오차가
  max(2.5px, 몸통 길이의 1.5%)를 넘으면 버린다. 4/9개 이상 경계쌍과
  연속 폭 안정성을 요구한다. 이웃 인물의 관절에 닿는 경계도 버린다.
- 엄격하게 좌우 각각 한 개만 존재하는 원본 선의 경계쌍은 별도 폴백으로 허용한다.
  검색창이 crop 밖이라는 이유만으로 crop 안의 정상 윤곽까지 버리던 오류를 수정했다.
  내부 근육선·여러 후보 경계가 있는 단면은 이 폴백으로 강제 선택하지 않는다.
- 초록 마스크는 관측된 단면 리본이다. 빨간 마스크는 가림/헐렁한 옷이 있는
  **관절 주변 제외 영역**이며 가려진 신체의 정확한 경계가 아니다.
  amodal 신체 복원·의상 아래 몸 추정·숨은 부위 채우기는 수행하지 않는다.
- `RoughBodyPipeline`은 위 단계를 연결한다. 소유권이 불확실하면 원격 판독과
  분할 호출을 선행 차단한다. 얼굴/피부 매끈함을 아동 체형의 근거로 쓰지 않는다.

## 같은 포즈·시점의 의미와 검증

`eval_body_rough_projection.py prepare`가 기존 Top-2 포즈 각각을
기존 `refine_bvh_v2(..., refine_mode='aggressive')`로 한 번 미세조정한다.
기존 selector·구조 안전·실패 시 베이스 복구 조건을 그대로 사용한다.
이 공통 BVH 하나와 카메라 하나를 20개 몸 모두에 동일하게 적용한다.
체형마다 포즈·시점·2D warp를 따로 최적화하지 않는다.

`render_body_pose_projections.py`는 확정 자산 남성 0.7.2 9종·여성 0.3.4 11종의
FBX를 실제로 retarget한 뒤, 9부위의 표면 재질 ID와 깊이 가림을 렌더한다.
정면/측면/3/4 등 검색 시점을 고정한다. 렌더에서 다른 신체 부위에 가려진
경계는 측정하지 않는다. 모든 후보에 공통으로 보이는 **동일 부위·동일 단면**만
비교하고 각 부위에 동일 가중치를 준다. 후보에 없는 부위를 점수 0으로 넣지 않는다.

고정된 공통 포즈는 러프의 정확한 3D 복원을 뜻하지 않는다. 방향 오차 중앙값
20도 이하, 최대 30도 이하를 요구한다. 길이가 달라도 방향이 같으면 통과하므로
체형 비례 차이를 포즈 오류로 일괄 차단하지 않는다. 정상 상반신은 몸통+팔에서
3부위·2영역 조건을 채우면 비교할 수 있으며 보이지 않는 다리를 요구하지 않는다.

최소 3부위·2영역, 1·2위 차이 > max(0.025, 4px/몸통 길이), 부위 하나씩 제거해도
동일 1위, 두 포즈 가설의 일관성을 요구한다. 이 임계값은 개발 가드이며
보정된 확률·확률 임계값이 아니다. 허용된 모든 경우도 진단용 review_candidate일 뿐
`selected_body_id=null`, `auto_apply=false`를 유지한다.

입력 crop, FBX, BVH, 투영 이미지, 렌더 코드, retarget 코드, 포즈·시점·공통 카메라,
20종의 완전성/중복을 검사한다. 코드가 바뀐 렌더 캐시는 재사용하지 않는다.
이 renderer는 frozen FBX의 진단용 LBS/retarget 경로이며 제품의 포즈별 surface QA나
전체 converter 보정 경로를 대체하지 않는다(`production_pose_qa=false`).

## 실제 사례: 131040:p0

| 공통 포즈 / 시점 | 방향 오차 중앙값 / 최대 | 점수 1위 | 1·2위 차이 | 윤곽 오차 하한 |
|---|---:|---|---:|---:|
| Sprint / three_quarter | 13.14° / 42.83° | male SOFT | 0.002379 | 0.040602 |
| Skateboarding / side | 10.68° / 13.15° | female TOON64 | 0.000701 | 0.040602 |

원본 Sprint 후보의 전완 방향 오차 85.82°는 공통 미세조정 후 42.83°가 됐지만
여전히 포즈 게이트를 넘는다. Skateboarding 후보는 55.77°→10.88°로 줄어
포즈 게이트를 통과한다. 그래도 1·2위 차이가 너무 작고 부위를 빼면 1위가 바뀐다.
이 순위는 추천 결과가 아니며, SOFT/TOON64를 실제 인물의 체형으로 판정하지 않는다.
순수 선화 `124702:p0`, `2.16.04:p0`은 노출 부위는 있으나 안정적인 분할에 실패했다.

## 재현

선택 의존성은 SAM 2 공식 저장소의 커밋
`2b90b9f5ceec907a1c18123530e92e794ad901a4`, torch 2.5.1 / torchvision 0.20.1,
hydra-core / iopath와 기존 Pillow/OpenCV다. 서버 기본 requirements에는 추가하지 않았다.
[공식 구현](https://github.com/facebookresearch/sam2)을 분리된 실험 환경에 설치한다.
체크포인트는 공식 `092824/sam2.1_hiera_tiny.pt`, SHA-256:
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.

```bash
# protocol.cases: case_id, crop/hash, detector points/mask, other_points,
# ownership_verified, poses[].{pose_id,view,bvh_path,pose_sha256}
# visibility.json: {people:{case_id:{parts,ownership_ambiguous}}}
python scripts/eval_body_rough_projection.py observe \
  --protocol protocol.json --visibility visibility.json \
  --checkpoint sam2.1_hiera_tiny.pt \
  --checkpoint-sha256 7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69 \
  --out run
# 새 Gemini 판독은 --visibility 대신 --allow-remote와 GEMINI_API_KEY를 명시한다.
python scripts/eval_body_rough_projection.py prepare \
  --observations run/observations.json --catalog config/body_morphology.v1.json \
  --fbx-root /path/to/package-minimal/fbx --out run/poses
blender -b --python scripts/render_body_pose_projections.py -- \
  --jobs run/poses/jobs.json --out run/projections
python scripts/eval_body_rough_projection.py compare \
  --observations run/observations.json --jobs run/poses/jobs.json \
  --projections run/projections/index.json --out run/results.json
```

최종 로컬 자료는 `artifacts/body-matching/2026-10-17-rough-projection/validated/`.
`REPORT.html`에 28명의 원본, 마스크, 제외 영역, 단면, 보류 사유, 전체 20종 점수가 있다.
`ARTIST_LABELS_TEMPLATE.json`은 28명과 crop 해시를 고정한 미라벨 양식이다.
raw 이미지·가중치·Gemini 원문·FBX·키는 커밋하지 않았다.
요약과 해시만 [ROUGH_PROJECTION_METRICS.json](ROUGH_PROJECTION_METRICS.json)에 보관한다.

## 회귀와 남은 조건

- 신규 29개: 의상/가림/스틱, 원본 경계 없는 마스크, 이웃 인물 소유권,
  crop 밖 검색창, 정상 반신, 길이 차이와 방향 오류 구분, 근소한 동점,
  부위 제거 불안정, 투영 자산/시점/카메라/코드/이미지 해시 변조와 후보 누락.
- CI와 같은 `pytest tests --ignore=tests/converter`: **872 passed, 8 skipped,
  28 subtests passed**. 기존 BFF/default 계약 포함. 스모크 **54/54**, compileall 통과.
  Blender 전용 converter 테스트는 일반 Python 수집 대상이 아니다.
  새 renderer는 별도로 실제 Blender에서 최신 20종 × 2포즈를 실행했다.
- 브라우저에서 28개 카드, 깨진 이미지 0개, 인물 필터, 모바일 가로 넘침 없음을 확인했다.

다음 검증은 (1) 실제 선화의 소유권·가시 윤곽 라벨, (2) 몸통 단축/원근을 포함한
포즈·시점 적합도, (3) 작가의 허용 프리셋 집합을 갖춘 분리 평가셋이 필요하다.
현 단계에서 게이트를 낮춰 자동 적용 수를 늘리는 것은 세부 체형 정확도 개선이 아니다.
