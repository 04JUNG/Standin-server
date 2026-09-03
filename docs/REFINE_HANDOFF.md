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

> ⚠ **아래 스케치는 §3 4단계로 무효가 됐다(2026-08-11).** `refined_handle`·
> `/refined/{handle}/bvh`·`REFINE_DIR`은 전부 제거됐다. 조정본은 `POST /refine` 응답의
> `bvh` 본문으로만 나가고 BFF가 S3에 보관한다. 이 작업을 실제로 할 때는 handle이 아니라
> **BFF의 `refined_artifacts`를 기준으로** 다시 설계해야 한다. 아래 코드 블록은 당시 의도를
> 남겨 둔 기록이다.

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

## 3. 결정 — 조정본을 `/refine` 응답에 실어 보낸다 (2026-08-05 확정)

조정본은 **refine을 처리한 인스턴스의 로컬 디스크**(`{DATA_DIR}/{REFINE_DIR}`)에 있다.

| 문제 | 언제 터지나 |
|---|---|
| 태스크 2개 이상이면 `POST /refine`과 `GET /refined/...`가 다른 인스턴스에 떨어져 404 | 다중 태스크 |
| 태스크가 교체되면 조정본이 사라짐 | ECS 롤링 배포·헬스체크 실패 |
| 캐시가 무한히 쌓임 | 장기 운영 |

### 왜 "단일 태스크"로는 부족했나

원래 MVP 결정은 "추론 서버를 단일 태스크로 띄운다, 코드 변경 0"이었다. **운영에서 공짜가
아니었다.**

`desiredCount=1`만으로는 부족하다. ECS 롤링 배포는 기본적으로 새 태스크를 먼저 띄우므로
구·신 태스크가 잠시 공존하고, Cloud Map이 두 주소를 모두 돌려준다. 그래서 실제로 한 태스크만
존재하게 하려면 `minHealthyPercent=0`(교체 후 기동)이 필요했는데 그 대가가 컸다.

- **배포 실패가 곧 장애가 된다.** 구 태스크를 먼저 내리므로 새 이미지가 healthy가 되지
  못하면 서비스할 태스크가 없다. 이 서버는 포즈 라이브러리나 VLM 키가 잘못되면 **의도적으로
  기동을 거부**하도록 만들어져 있어서(`runtime_guard`, `_ensure_db`) 실제로 밟을 수 있는 경로다.
- **배포 중 수십 초~2분 중단**이 매번 발생한다.
- AWS가 `maximumPercent<=100`을 **Availability Zone 재분산이 켜진 서비스에서 거부**해
  프로덕션 배포가 400으로 실패하고 CloudFormation이 롤백했다. 재분산을 끄는 것으로 우회했다.

인프라는 현재 `refineEnabled` 조건부로 이 제약을 refine을 켤 때만 걸도록 해 두었다. 즉
**refine을 켜는 순간 위 비용이 되살아난다.**

### 채택안 — 바이트 인라인

`POST /refine` 응답에 조정 BVH 본문을 함께 실어 **두 번째 요청을 없앤다.**

```text
before   POST /refine → (로컬 디스크에 씀) → GET /refined/{handle}/bvh → BFF가 S3 저장
after    POST /refine → 응답에 본문 포함 → BFF가 S3 저장
```

두 번째 요청이 없으면 태스크 친화성 요구 자체가 사라진다. 태스크 수·배포 전략·AZ 재분산
어느 것도 신경 쓸 필요가 없어지고, 인프라의 조건부 분기를 전부 지울 수 있다.

**크기 걱정은 없다.** `refine_bvh`는 `write_single_frame_bvh`로 **단일 프레임** BVH를 쓴다
(`src/refine.py`). HIERARCHY 블록 + MOTION 한 줄이라 JSON 문자열로 실어도 부담이 되지 않는다.

공유 스토리지(S3 직접 쓰기)는 채택하지 않았다. 추론 태스크에 S3 쓰기 권한과 버킷 설정이
붙는데, 지금 구조에서는 **BFF가 유일한 S3 writer**인 편이 권한 경계가 단순하다.

> §3의 원래 우려("동원이 URL 다운로드를 전제하면 양쪽 지원이 필요")는 해소됐다. 조정본을
> 소비하는 것은 BFF 하나뿐이고, 클라이언트는 BFF의 공개 export URL만 쓴다. 추론 서버의
> `/refined/{handle}`를 외부에서 직접 받아가는 경로는 없다.

### 계약 변경

`RefineResponse`에 필드 하나를 **추가만** 한다.

```python
# api/models.py::RefineResponse
bvh: Optional[str] = Field(
    None, description="조정본 BVH 본문(LF 개행). refined=true일 때만 채운다. "
                      "소비자는 이 값을 받아 자기 저장소에 보관한다.")
```

- `bvh_url`은 **그대로 둔다.** 구 소비자가 계속 동작해야 하고, `refined=false`일 때는
  여전히 베이스 경로를 가리키는 의미가 있다.
- 로컬 파일 쓰기와 `GET /refined/{handle}/bvh`도 당장은 유지한다. 순차 배포 중 구 BFF가 쓴다.
- **개행은 LF로 고정한다.** 단, 위험은 *읽기*가 아니라 *쓰기* 쪽이다 (2026-08-07 리뷰로 정정).
  - HIERARCHY는 이미 LF로 정규화된다. `hierarchy_text()`가 universal newlines로 읽어
    `splitlines()` + `"\n".join()`으로 재조립하므로(`src/bvh.py`), 원본 BVH가 CRLF여도 이
    시점에 LF가 된다. MOTION 블록은 f-string으로 직접 만든다. 즉 **응답에 실리는 `bvh`
    문자열은 플랫폼과 무관하게 항상 LF다.**
  - 반면 `write_single_frame_bvh()`의 `open(out_path, "w")`는 `newline=`을 지정하지 않아
    `\n`이 `os.linesep`으로 번역된다. Linux 컨테이너에서는 LF지만 Windows에서 돌리면 디스크
    파일이 CRLF가 된다. 1~3단계 동안 두 경로가 공존하므로, 신 BFF(`bvh` 필드, 항상 LF)와
    구 BFF(`GET /refined/{handle}/bvh`, 서버 플랫폼 의존)가 **서로 다른 바이트**를 받을 수
    있다. 동원의 CSP 축 보정·드래그가 걸린 부분이라 조용히 틀리면 찾기 어렵다.
  - → 1단계에서 `newline="\n"`을 명시해 두 경로의 바이트를 일치시킨다.

### 소비자 쪽 (`Standin-app-server`)

`src/refine/service.ts`가 두 단계를 한 단계로 줄인다.

```ts
// upstream.bvh가 있으면 그대로 저장한다. 없으면(구 추론 서버) 기존 경로로 폴백.
const bytes = upstream.bvh
  ? new TextEncoder().encode(upstream.bvh)
  : new Uint8Array(await (await deps.fetchUpstreamPath(upstream.bvh_url)).arrayBuffer());
await deps.putRefinedBvh(objectKey, bytes);
```

`fetchUpstreamPath`와 `RefineDeps`의 해당 항목은 폴백 제거 단계까지 남긴다.

### 배포 순서

각 단계가 **단독으로 안전**해야 한다.

| # | 저장소 | 내용 | 왜 안전한가 |
|---|---|---|---|
| 1 | Standin-server | `bvh` 필드 추가 + `write_single_frame_bvh`에 `newline="\n"` | 순수 추가. 구 BFF는 무시한다. 개행 고정은 두 경로의 바이트를 맞춘다 |
| 2 | Standin-app-server | `bvh` 우선, 없으면 폴백 | 구·신 추론 서버 모두 동작 |
| 3 | Standin-infra | `refineEnabled` 삼항 제거 → 항상 `100/200` + AZ 재분산 | 이 시점엔 두 번째 요청이 없다 |
| 4 | server / app-server | `/refined/{handle}`·로컬 쓰기·폴백 제거 | 정리 |

**1~3은 2026-08-11에 프로덕션 배포 완료.** 배포 직후 실제 refine 1건으로 확인했다 —
`POST /refine` 뒤에 `GET /refined/...`가 따라붙지 않고, BFF에 `refine_applied`가 남았다.

**3번을 2번보다 먼저 하면 안 됐다.** 구 BFF가 아직 두 번 요청하는 상태에서 무중단 배포로
되돌리면 조정본 404가 난다. 제약은 머지가 아니라 **배포**에 걸린다는 점에 주의 — develop
머지로는 아무것도 배포되지 않고, `main` 푸시가 자동 배포를 트리거한다(인프라는 수동
`cdk deploy`).

4번을 하면 "캐시가 무한히 쌓임" 문제도 함께 사라진다. handle 기반 멱등 캐시가 없어지지만,
BFF의 `refined_artifacts` PK(`job_id, person_index, candidate_id`)가 같은 선택의 재호출을
막으므로 충분하다.

**4번 적용됨(2026-08-11).** 서버 쪽 변경:

- `GET /refined/{handle}/bvh` 라우트 제거. 사이드카 JSON과 `_refine_handle()`도 함께 제거
- `refine_bvh(out_path=None)`이 **파일을 쓰지 않는다.** 본문은 `RefineResult.bvh_text`로
  돌려주고, 추론 API는 이 경로로 호출한다. 평가·진단 스크립트는 계속 `out_path`를 준다
- `REFINE_DIR` 설정 제거
- `bvh_url`은 refined 여부와 무관하게 항상 `/pose/{id}/bvh`(베이스)
- 최종 결과의 front PNG는 같은 `/refine` 응답의 `thumbnail`에 base64로 인라인한다.
  기존 후보와 같은 `warm-mannequin-v1` 렌더러를 쓰며 PNG도 로컬에 저장하지 않는다

4번의 전제는 리뷰에서 확인됐다(2026-08-07). `/refined/{handle}/bvh`를 HTTP로 읽는 코드는
없고, 평가·진단 스크립트(`run_batch_pipeline.py`, `eval_refine_batch.py`, `diag_refine_3d.py`,
`run_skeleton_pipeline_bundle.py`)는 전부 `src.refine.refine_bvh`를 직접 호출해 자체
`out_path`에 쓴다 — 로컬 쓰기 제거와 무관하다.

### 검증

- **server**: `refined=true`면 `bvh`가 비어 있지 않고 `refined=false`면 `None`인지. 계약
  fixture에 추가한다. 두 경우 모두 `thumbnail`이 유효한 256×256 PNG인지 확인한다.
- **app-server**: `bvh`가 있으면 `fetchUpstreamPath`를 **호출하지 않고** 저장하는지, 없으면
  폴백하는지. 기존 `RefineDeps` 주입 구조로 DB·S3 없이 검증할 수 있다.
- **통합**: 추론 태스크를 교체한 뒤에도 export가 성공하는지(E2E-08). 이 변경의 목적이므로
  staging에서 반드시 확인한다.

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
