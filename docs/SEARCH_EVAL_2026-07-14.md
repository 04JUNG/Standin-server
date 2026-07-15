# 검색 정성평가 리포트 (4순위) — 2026-07-14

러프 콘티 → RTMPose → kNN Top-5 파이프라인을 **실 데이터**로 처음 end-to-end 검증한 기록.
목적: "검색 결과가 고쳐 쓸 만한가"를 사람 눈으로 판정 + 막힌 지점 정리.

## 1. 셋업 (이번에 구축된 것)

| 항목 | 값 |
|------|-----|
| 라이브러리 | Mixamo FBX → `mixamo_fbx_to_bvh.py`(apex 프레임) → BVH **77개** |
| DB | `BVH_DIR=data/bvh python scripts/build_db.py` → `data/poses.db` (**77 poses / 308 projections**, view 4종) |
| 쿼리 추출 | `eval_search.py --image` → RTMPose Body(`rtmpose-x_simcc-body7`, 176MB, onnxruntime CPU) |
| 실행 파이썬 | **`py -3.12`** (numpy/matplotlib/rtmlib/onnxruntime/opencv 설치됨). `python`=msys2(파이썬, pip 없음) 주의 |
| 평가셋 | `Downloads/library_test/conti` 14컷 (panel_*.png 6 + 스크린샷 8) |

## 2. 결과 요약 (14컷 Top-1)

| # | 컷 | Top-1 | dist / ~sim | 판정 |
|---|----|-------|------|------|
| 1 | panel_0 | Kneel_02 | 0.324 / 0.68 | ○ |
| 2 | panel_2 | Hands Forward Gesture (¾뷰) | 0.208 / 0.79 | ○ |
| 3 | panel_3 | Sitting Idle | 0.589 / 0.41 | △ |
| 4 | panel_4 | Sitting Idle | 0.486 / 0.51 | ○ |
| 5 | panel_5 | Quick Formal Bow (side) | 0.649 / 0.35 | △ |
| 6 | panel_6 | Sitting Idle | 0.408 / 0.59 | ○ |
| 7 | 스크린샷 …540 | Stand_01 | **1.230 / 0.00** | ✗ 추출 실패 |
| 8 | 스크린샷 …607 | Seated Idle | 0.560 / 0.44 | ○ |
| 9 | 스크린샷 …623 | Sitting Idle | 0.511 / 0.49 | ○ |
| 10 | 스크린샷 …629 | Standing Greeting | 0.141 / 0.86 | ◎ |
| 11 | 스크린샷 …637 | Sitting Idle | 0.589 / 0.41 | △ |
| 12 | 스크린샷 …648 | Sitting Idle | 0.514 / 0.49 | ○ |
| 13 | 스크린샷 …702 | Quick Formal Bow | 0.197 / 0.80 | ◎ |
| 14 | 스크린샷 …720 | Kneel_01 | 0.168 / 0.83 | ◎ |

결과 이미지: `Downloads/library_test/conti/eval/eval_01~14.png` (컷별 [쿼리|Top-5] 패널)

판정 기호: ◎ 바로 쓸 만 / ○ 고쳐 쓸 만 / △ 애매 / ✗ 실패

## 3. 판정 — 검색 자체는 동작한다

- **쿼리 스켈레톤이 깨끗하면 변별이 됨**: 앉기→Sitting, 무릎→Kneel, 인사→Greeting, 절→Bow로 카테고리가 맞게 걸린다.
- 시각 확인:
  - `eval_14`(무릎 꿇은 쿼리) → **#1 Kneel_01** 정확. dist=0.168.
  - `eval_10/13`(인사/절) → sim 0.80~0.86, 상위 후보가 동일 계열.
- **거리 스프레드 회복**: 0.14 ~ 1.23으로 넓게 퍼짐 → sim이 실제 유사도를 반영.
- **Sanity check 통과**: 라이브러리 BVH 자기 자신을 쿼리로 넣으면 dist=0.000, #1 자기 자신 (피처 공간 대칭성 정상).

> ⚠️ 앞선 "모든 거리가 0.014로 붕괴" 현상은 **버그가 아니라** *서기 쿼리 × 서기-편중 라이브러리*의 특수 케이스였음. 다양한 콘티에서는 재현되지 않음.

## 4. 발견된 문제 (우선순위)

### P1 — cv2 한글 경로 버그 (실제 코드 버그, 실사용 치명)
- 증상: 한글 파일명(`스크린샷…png`) 8장 전부 `AttributeError: 'NoneType' object has no attribute 'shape'`.
- 원인: `src/pose.py::RTMPoseModel.estimate`의 `cv2.imread(image)`가 **유니코드 경로에서 None 반환**(Windows OpenCV 알려진 한계).
- 현재 우회: ASCII 이름으로 복사 후 실행.
- 권장 수정(한 줄): 
  ```python
  # cv2.imread(image) 대체
  img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
  ```
- 영향: 실제 작가 파일명은 한글이 흔함 → **반드시 고쳐야 함**.

### P2 — 라이브러리 다양성/밀도 부족
- 77개가 서기·앉기 중심으로 편중. 서기 계열 쿼리에서 후보가 서로 잘 안 갈림(△ 케이스 다수).
- △ 판정(#3,5,11)과 낮은 sim(0.35~0.41)이 몰린 원인.
- 대응: 격투·눕기·크게 벌린 동작 등 **실루엣이 뚜렷이 다른 포즈 추가**. (기존 진단문서의 "라이브러리 밀도" 방향과 일치)

### P3 — RTMPose 추출 실패 컷 (#7)
- 쿼리 스켈레톤이 깨짐(머리·팔 분리, 몸통이 삼각형). 다중 인물/크롭/잘림 추정.
- 시스템 동작은 **정상**: dist=1.230(sim 0.00)으로 "매칭 없음"을 정직하게 냄 → 저신뢰 폴백 신호로 활용 가능.
- 대응: 입력 컷 품질 확인, 검출 박스(YOLO/RTMDet) 단계 점검.

### P4 — 대표 프레임(apex) 편중 가능성
- `mixamo_fbx_to_bvh.py`가 클립당 비슷한 순간(apex)을 뽑아 서기 편중을 키웠을 수 있음.
- 대응: `bvh_contact_sheet.py`로 77개 실제 포즈를 컨택트 시트로 훑어 편중/중복 확인.

### P5 — 환경/실행 주의
- `python`이 pip 없는 msys2 파이썬을 가리킴 → 반드시 **`py -3.12`** 사용.
- 세션 작업 폴더가 Windows 패키지 앱 가상화 경로라 **외부 프로세스(blender.exe)가 접근 불가** → Blender 배치는 `C:\tmp`에 스크립트·입출력을 두고 실행함(프로젝트 내부 Python 실행은 무관).

## 5. 다음 액션 (권장 순서)

1. **[P1] cv2 한글 경로 버그 수정** — 한 줄, 즉시. 한글 컷 바로 실행 가능해짐.
2. **[P4] 컨택트 시트로 라이브러리 편중 눈으로 확인** — 데이터 문제 범위 확정.
3. **[P2] 라이브러리 다양화** — 실루엣이 다른 포즈 보강 후 DB 재빌드, 14컷 재평가.
4. **[P3] 추출 실패 컷 원인 분리** — 입력/검출 단계 점검.

## 6. 재현 커맨드

```powershell
# DB 재빌드(실 BVH)
$env:BVH_DIR="data/bvh"; py -3.12 scripts/build_db.py

# 단일 컷 평가
py -3.12 scripts/eval_search.py --image <cut.png> --db data/poses.db --topk 5 --out eval.png

# 자기 쿼리 sanity check
py -3.12 scripts/eval_search.py --from-bvh "data/bvh/Quick Formal Bow_01.bvh" --db data/poses.db --topk 5 --out eval_self.png

# 진단: 상체 가중 / 전체 순위 / 특정 포즈 순위
py -3.12 scripts/eval_search.py --image <cut.png> --upper-weight 3 --rankall --mark Kneel
```
