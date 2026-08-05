# refine 인수인계 — 이 브랜치가 **안 한 것**과 다음 사람이 할 것

> 브랜치: `feat/refine-pipeline` · 작성: 2026-07-31 (도원)
> 한 줄: **추론 서버 안쪽은 끝났다. 조정본이 동원까지 가는 배선은 아직이다.**

---

## 0. 왜 이 문서가 있나

`POST /refine`은 동작하고 조정본 BVH도 정상이다. 그런데 **작가가 실제로 쓰는 경로**는
`/analyze → 선택 → /export-order → 동원 → CSP`인데, `/export-order`는 refine을 모른다.

```python
# api/app.py::export_order  (현재)
bvh_url=f"/pose/{sel.pose_id}/bvh"     # ← 항상 베이스. 조정본은 여기로 안 흐른다.
```

즉 **이 브랜치만 머지하면 refine은 막다른 길이다.** 작가가 조정본을 받아도
내보내기는 베이스를 가져간다. 아래 §1을 하지 않으면 refine은 켜지지 않은 것과 같다.

> 이 사실을 PR 본문에 반드시 적을 것. "구현했으니 됐다"로 읽히면 시연에서 터진다.

---

## 1. 남은 작업 A — `/export-order`에 조정본 연결 (담당: 내보내기 쪽)

### 무엇을

작가가 refine을 받았으면 그 결과가 주문서에 실려야 한다. **재계산은 하지 않는다** —
클라이언트가 `/refine` 응답에서 받은 `bvh_url`(또는 handle)을 들고 있다가 그대로 넘긴다.
서버가 "이 컷의 마지막 refine"을 기억하면 무상태 원칙(`DECISIONS.md` 결정 4)이 깨진다.

### 계약 변경 (schema_version `1.0` → `1.1`, 전부 추가 필드라 하위호환)

```python
# api/models.py
class ExportSelection(BaseModel):
    person_index: int
    pose_id: str
    view: str
    refined_handle: Optional[str] = None   # ← 추가. /refine 응답 bvh_url의 handle

class ExportItem(BaseModel):
    ...
    bvh_url: str                            # 조정본이 있으면 /refined/{handle}/bvh
    base_bvh_url: Optional[str] = None      # ← 추가. 항상 /pose/{pose_id}/bvh
    refined: bool = False                   # ← 추가
```

```python
# api/app.py::export_order  안에서
refined = bool(sel.refined_handle) and os.path.exists(
    os.path.join(REFINE_DIR, f"{sel.refined_handle}.bvh"))
items.append(ExportItem(
    ...,
    bvh_url=(f"/refined/{sel.refined_handle}/bvh" if refined
             else f"/pose/{sel.pose_id}/bvh"),
    base_bvh_url=f"/pose/{sel.pose_id}/bvh",
    refined=refined,
))
```

### 왜 이 형태인가

- **`base_bvh_url`을 항상 같이 준다.** 동원 쪽에서 조정본이 이상하면 베이스로 즉시
  되돌릴 수 있다. 시연 중 탈출구가 하나 더 생긴다.
- **`refined_handle`이 없거나 파일이 없으면 조용히 베이스**로 떨어진다. 404를 내지
  않는다 — 내보내기가 실패하는 것보다 베이스라도 나가는 게 낫다("좋아지거나, 그대로").
- `pose_id`는 그대로 유지된다. 동원의 파일명·소재 폴더 규칙은 계속 `pose_id`를 쓰면 된다.

### 같이 고칠 문서 (`COLLABORATION.md` §6 규칙)

- `docs/EXPORT_CONTRACT.md` — §1 요청 표에 `refined_handle`, §2 응답 예시에
  `base_bvh_url`·`refined`, §4에 "조정본도 동일하게 소비"를 추가.
- `docs/API_CONTRACT.md` §2 표 — `/export-order` 행 갱신.

### 동원에게 확인받을 것

1. 조정본 URL이 `/refined/{handle}/bvh`로 와도 되는지(파일명은 `{pose_id}.refined.bvh`로 내려간다).
2. `refined: false`일 때 아무 처리도 하지 않아도 되는지(그냥 베이스다).
3. 주문서에 `base_bvh_url`을 넣는 게 도움이 되는지, 아니면 소음인지.

---

## 2. 남은 작업 B — 뷰어가 조정본을 다시 로드 (담당: 클라/뷰어 쪽)

**시연에서 가장 위험한 항목이다.**

지금 뷰어는 Top-5(=베이스)를 보여주고, 작가는 그걸 보고 고른다. 서버는 고른 뒤에
조정한다. 그래서 **작가가 한 번도 본 적 없는 포즈가 CSP로 나간다.**
30분 라이브에서 "내가 고른 거랑 다른데요"가 나오면 회복이 안 된다.

- 서버가 추가로 줄 건 없다 — `/refine` 응답의 `bvh_url`이 이미 있다.
- 뷰어가 **선택 직후 그 URL을 다시 로드해서 표시**하면 끝난다.
- `refined: false`면 베이스 URL이 오므로 뷰어는 두 경우를 구분할 필요가 없다.

권장 UI: 조정됐을 때 작은 배지 하나("러프에 맞춰 조정됨"). 게이트에 걸렸으면
`reason`을 근거로 "라이브러리에 비슷한 게 없어 조정하지 않았습니다"를 보여준다.
**정직한 실패는 신뢰를 얻고, 침묵은 잃는다.**

---

## 3. 보류 결정 — 조정본 파일의 수명과 배포 (팀 합의 필요)

조정본은 **refine을 처리한 인스턴스의 로컬 디스크**(`{DATA_DIR}/{REFINE_DIR}`)에 있다.

| 문제 | 언제 터지나 |
|---|---|
| 태스크 2개 이상이면 `POST /refine`과 `GET /refined/...`가 다른 인스턴스에 떨어져 404 | ALB 뒤 다중 태스크 |
| 태스크가 교체되면 조정본이 사라짐 | ECS 롤링 배포·헬스체크 실패 |
| 캐시가 무한히 쌓임 | 장기 운영 |

**MVP 결정: 추론 서버는 단일 태스크로 띄운다.** 코드 변경 0, 시연 규모에 충분.
아래는 나중 선택지(지금 고르지 않는다):

- **바이트 인라인** — `/refine`이 BVH 내용을 응답에 실음. 무상태가 되지만 동원이
  URL 다운로드를 전제하면 양쪽 지원이 필요. (`EXPORT_CONTRACT` §4 확인필요 ①과 같은 논점)
- **공유 스토리지(S3/EFS)** — 제대로 된 해법. 인프라 작업이 붙는다.
- **TTL 청소** — 어느 쪽을 고르든 필요.

---

## 4. 이 브랜치가 **한 것** (참고)

| 영역 | 내용 |
|---|---|
| 코어 | `src/refine.py` — 팔·다리 8관절, 각도 손실, scipy/numpy 백엔드, 안전 게이트 10종 |
| 리팩터 | `bvh.coco17_from_fk` 분리 · `bvh.write_single_frame_bvh`(HIERARCHY 원문 보존) · `library.pose_to_feature`(색인·refine 공유) |
| API | `PersonOut.keypoints/scores` · `POST /refine` · `GET /refined/{handle}/bvh` |
| 도구 | `eval_refine.py` · `eval_refine_batch.py` · `refine_top5.py` |
| 테스트 | 스모크 8건 추가(게이트 동작·BVH 유효성·피처 대칭성) |

경계 밖은 건드리지 않았다:

- **동원의 CSP 미러링·축 보정**(`DECISIONS.md` 결정 3) — 영향 없음. 조정본은 HIERARCHY가
  원본과 **바이트 단위로 동일**하고 MOTION 회전값만 다르다. 관절명·OFFSET·채널 순서·축
  규약이 그대로라 보정 로직을 손댈 이유가 없다.
- **BFF** — `/v1` 매핑에 엔드포인트 2개 추가만 필요(`BFF_DESIGN.md`). refine은 ~1s라
  Job으로 감쌀 필요 없이 동기 프록시로 충분.
- **얽힘 세트** — `set_id`가 있는 포즈는 refine을 **스킵**한다(`reason=entangled_set`).
  각자 돌리면 두 사람이 맞물리던 정합이 깨지는데 BVH는 상대 위치를 안 실어 되돌릴 수 없다.
  세트 refine은 세트 전체를 함께 푸는 별도 과제.

---

## 5. 머지 전 체크리스트

- [ ] `python tests/test_smoke.py` 통과 (현재 21/21)
- [ ] `python -m compileall src api scripts` 통과
- [ ] `data/` · `.env` · `*.onnx` 미포함
- [ ] PR 본문에 **"§1을 하기 전엔 조정본이 동원까지 가지 않는다"** 명시
- [ ] §1(export 배선) 이슈 생성 후 링크
- [ ] §2(뷰어 재로드) 이슈 생성 후 링크
- [ ] 실 러프 14컷 눈검수(`eval_refine_batch.py`) — `REFINE_DESIGN.md` §6-1
