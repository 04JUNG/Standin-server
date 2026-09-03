# 아키텍처 결정 — DB · 라이브러리 저장 · 동원 핸드오프

> 도원 담당: **VLM 분석 → 검색** Python 추론 서버(FastAPI). 앱 서버(친구들)가 HTTP로 호출.
> 이 문서는 세 결정을 확정한다: ① DB ② 포즈 라이브러리 저장 형태 ③ 동원(내보내기)에게 전달 방식.
> ⚠ 표시는 **동원/팀과 확인 필요**한 항목.

---

## 결정 1 — DB: SQLite (MVP), 인터페이스로 교체 가능

**선택: SQLite.** 이유:
- 무설치·단일 파일 → 백업·공유·버전관리가 파일 하나. 라이브러리를 "아티팩트"로 넘기기 쉬움.
- 태그(shot/action/relationship)가 진짜 컬럼 → 사전필터가 SQL `WHERE`.
- MVP 규모(포즈 200~500 × 뷰 4 ≈ 800~2000 벡터)에선 **kNN을 메모리 브루트포스로 해도 즉시**(cosine 2000×34는 마이크로초). 벡터 전용 DB가 아직 불필요.

**안 고른 것 / 승격 경로**
- **pgvector(Postgres)**: 앱팀이 이미 Postgres를 돌리면 인프라 공유 이점. 도원 서비스 단독 MVP엔 운영 부담이 과함.
- **LanceDB(임베디드 벡터DB)**: 서버리스 + 필터+벡터 네이티브. 처음부터 벡터 스케일을 원하면 매력적. 지금은 SQLite로 충분하고, 스케일 시 `repo.py`만 교체.
- 규칙: **검색 인터페이스(`search.knn`)는 불변**, DB는 `src/repo.py` 뒤에 숨긴다. 스케일이 커지면 이 파일만 바꾼다.

**경로**: `DB_PATH`(env, 기본 `data/poses.db`). ⚠ 동기화 폴더(드롭박스/OneDrive 등)에 두면 SQLite 락 오류 → 로컬 디스크 경로 사용.

---

## 결정 2 — 포즈 라이브러리 저장 형태

> ✅ **확정(도원)**: ① 비교 벡터 = **2D 키포인트 벡터(34차원 정규화)**  ② 시점 = **뷰별 2D 투영을 미리 구워 저장**.
> 검토 후 기각: 관절각 벡터(구현비↑, MVP엔 이득 불확실) · 학습 임베딩(기성 모델 원칙 위반) · 쿼리 시 투영(쿼리 무겁고 복잡).
> 남은 구현 디테일 2개(아래 ★).

**핵심 원리(설계문서 §3·§12·§15)**: 그림에서 3D 복원(리프팅) 안 함. 3D 포즈(BVH)를
**여러 가상 카메라로 2D 투영**해 색인하고, 추출한 2D 스켈레톤과 같은 피처 공간에서 kNN.
그래서 원본 3D 1개 → 투영(view) 여러 개가 저장된다.

**스키마 (2 테이블)**
```
poses(pose_id PK, bvh_path, source, license, shot, action,
      relationship, set_id, set_role, meta_json)
      · bvh_path  = 원본 BVH 파일 위치(바이트를 DB에 안 넣음 — 파일이 곧 동원 산출물)
      · license   = CMU/Mixamo '재배포 금지' 조항 메모(우리는 추천 서비스라 OK지만 필드로 추적)
      · set_id/set_role = 얽힘 그룹(nullable). ⚠ BVH는 1인만 지원 → '2인 BVH' 없음.
        얽힘도 1인 BVH 여러 개를 set_id로 묶어 표현(상대 위치는 작가가 CSP에서).

pose_projections(id PK, pose_id FK, view, feature_blob, feature_version)
      · feature_blob   = 정규화 2D 스켈레톤 (17×2=34) float32 → bytes(BLOB)
      · feature_version= 정규화 규격 버전. features.normalize_skeleton이 바뀌면 +1 하고 재빌드.
                         (쿼리/라이브러리 정규화 불일치를 런타임에 잡는 안전장치)
```

**왜 이렇게**
- **피처는 벡터로, BVH는 파일로.** DB는 "무엇을·어떤 태그로·어디 BVH"만. 무거운 BVH 바이트를 DB에 넣지 않아 가볍고, 파일은 그대로 동원에게 줄 산출물.
- **feature_version**이 결정 4(불변식: 쿼리·라이브러리 동일 정규화)를 강제. 한쪽만 바꾸면 로드 시 에러.
- ⚠ **BVH는 1인만 지원(위치 미반영)** — 멘토링 회의록 확인. 그래서 '2인 BVH'는 없다.
  얽힘(hug/fight)은 **1인 BVH 여러 개를 set_id로 묶어** 저장. export는 인물 수만큼 1인 BVH item.

**★ 라이브러리 태그는 VLM이 부여(반자동 + 사람 검수) — 설계문서 §12-3단계**
쿼리와 **같은 VLM·같은 Controlled Vocabulary**로 라이브러리도 태깅해야 태그 공간이 일치한다(action='sitting' 끼리 매칭되려면 양쪽 태거가 같아야 함 — 피처 공간 대칭과 같은 이유).
단, 태그별로 출처가 다르니 구분한다:
- **action / shot → VLM**. 렌더한 포즈 이미지를 VLM에 넣어 태깅. Mixamo/CMU는 모션 이름(walk, sit…)이 있으니 **출처 라벨을 시드로** 쓰고 VLM으로 우리 어휘에 정규화·보정 후 **사람 검수**.
- **view → VLM 아님(결정론적)**. 라이브러리는 우리가 투영 카메라 각도를 알고 있으므로 view는 투영에서 그대로 확정. VLM view는 그림을 모르는 **쿼리 시점에만** 필요.
- **relationship → 등록 시 지정**. 1인 solo, 얽힘은 set_id로 묶인 1인 BVH들에 hugging/fighting 지정.
→ 즉 '무엇을(action/shot)'만 VLM 반자동, '어느 각도(view)·몇 명(relationship)'은 기하/메타에서 확정.

**★ 확정에 따른 실구현 디테일 2개**
- **키프레임 추출**: BVH는 애니메이션(여러 프레임). 정지 포즈 소재는 대표 1프레임, 모션 클립은 고정 간격 또는 수동으로 키프레임 몇 개만 뽑아 각각 1 pose로 등록. (전량 프레임 색인 금지 — 중복만 늘어남)
- **BVH 파일 위치**: DB엔 경로만(`data/bvh/{pose_id}.bvh`), 실체는 파일시스템. 스케일 시 오브젝트 스토리지(S3 등)로 경로만 교체. BFF는 base 선택에서 `/pose/{id}/bvh`로 이 파일을 받음(결정 3).

**실데이터 연결점(TODO)**: `library.load_bvh_pose()`에 BVH 파서(bvhio 등) + BVH 관절명→COCO17 매핑.
CSP 실험에서 '구조 기반 매핑' 확인됨(이름 관대) → 매핑 규칙 자유도 있음. 이후 `scripts/build_db.py`가 폴더 순회로 채움.

---

## 결정 3 — base BVH 제공과 최종 Converter 핸드오프

**패턴 A 유지: 라이브러리 base BVH의 단일 소스는 inference 서비스다.**

> 2026-08-27 V3.2 Converter 통합으로 downstream 책임을 갱신했다. `/pose/{id}/bvh`는 계속
> base 원본의 단일 소스지만, refine 성공 시 최종 입력은 inline `RefineResponse.bvh`다. BFF가
> 최종 바이트를 하나로 확정해 별도 내부 Converter API에 넘긴다.

```
POST /analyze  → CutResult {
    people: [ { index, box, tags, candidates: [
        { pose_id, view, distance, tags, rerank_score, bvh_url } ] } ]
}
GET  /pose/{pose_id}/bvh   → 라이브러리 BVH 원본(application/octet-stream)
```

흐름: 앱이 `/analyze`로 Top-K 후보를 받아 표시 → 작가가 1개 선택 → 선택적 `/refine` →
BFF가 base URL 응답 또는 refined inline 본문 중 최종 BVH 바이트를 확정 → 내부 `/convert` →
V3.2 FBX → BFF 저장·다운로드 → CSP 배치.

**책임 분리 (경계선)**
| 단계 | 소유 | 내용 |
|------|------|------|
| pose_id·후보·base BVH 제공 | **inference** | 검색 결과 + `/pose/{id}/bvh` |
| 최종 base/refined 바이트 확정·SHA lineage | **BFF** | inline refined 우선, 아니면 base GET |
| BVH→FBX rest/chain retarget·명시적 mirror | **Converter** | Blender 5.2 + 동결 V3.2 |
| FBX 저장·공개 다운로드 | **BFF** | conversion_id와 입출력 SHA 기록 |
| 소재 등록·다인 상대 위치·CSP 배치 | **CSP/작가** | Converter mirror를 반복하지 않음 |

Converter 도입 전 CSP가 맡았던 BVH 좌우 반전·다리/루트 retarget은 동결 V3.2 출력단으로
이동했다. MVP의 mirror는 Converter 요청에서 정확히 한 번 적용한다. CSP는 iPad/CSP 소재 등록과
배치를 확인하되 같은 BVH 수학을 다시 적용하지 않는다.

**왜 패턴 A인가 (vs BFF/Converter가 라이브러리 사본 보유)**
- 라이브러리 **단일 소스** 유지 → 두 서비스가 BVH 사본을 동기화할 필요 없음.
- inference는 후보와 base BVH, BFF는 최종 선택, Converter는 FBX 변환, CSP는 배치를 소유한다.

확정된 전달 방식:

1. base는 URL 응답 바이트, refined는 `/refine` inline 본문을 BFF가 받는다.
2. BFF는 최종 바이트를 multipart로 Converter에 업로드한다.
3. mirror는 Converter에서 한 번만 적용한다.
4. 다인은 인물별 독립 BVH→FBX이며, `set_id`는 묶음 메타다.

세부 계약은 `docs/FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md`를 따른다.

---

## 결정 4 — 앱 서버(BFF) 분리 vs FastAPI 단일 처리

> 배경: 클라이언트(`08_API_CONTRACT.md`)는 **비동기 Job + 토큰 인증 + `/v1`**을 가정하는데,
> 이 추론 서버는 **동기 `/analyze` + 무인증**이다. 그 간극을 누가 메우나가 이 결정이다.
> (`docs/API_CONTRACT.md` §0·§8-1에서 제기한 질문의 확정.)

**선택: 얇은 앱 서버(BFF)를 별도 계층으로 둔다. 이 추론 서버는 순수 추론으로 유지.**

```
[Tauri] ──> [앱 서버(BFF)] ──> [도원 추론 서버 = 이 저장소]
           인증·Job·기록·오류봉투     POST /analyze (동기·무인증·무상태)
           /v1·레이트리밋            GET  /pose/{id}/bvh · POST /export-order
```

**이유(우선순위)**
1. **런타임 성격이 다르다(핵심).** 추론=무거운 모델(rtmpose 176MB)·CPU/GPU 바운드·긴 처리·워커 적게. 인증/기록=가벼운 I/O·수평 확장. 한 프로세스에 섞으면 엉뚱한 걸 스케일하고(로그인 받자고 모델 N개), 추론 하나가 막히면 로그인도 같이 느려진다.
2. **팀 경계.** 이 저장소의 계약은 "입력=컷 이미지, 출력=`CutResult` 두 끝만 맞추면 병렬 개발"(`CLAUDE.md`). 인증·유저·Job을 넣으면 이 깨끗한 경계가 깨지고 모델 배포와 제품 배포가 커플링된다.
3. **보안.** 무인증 추론 서버를 앱 서버 뒤(내부망)에 둬 외부에 노출하지 않는다. **공개 엣지는 앱 서버 하나**가 인증·레이트리밋·오류봉투를 소유.
4. **클라가 이미 BFF 형태를 가정.** `endpoints.ts`(`/v1`)·`client.ts`(Bearer·오류 정규화) 대로면, 얇은 앱 서버가 `/analyze`·`{detail}`(추론 방언)을 클라 계약으로 **번역**하는 계층까지 겸한다.
5. **데이터 수명주기.** 추론 SQLite=포즈 라이브러리(읽기 위주·단일 소스). 유저·토큰·Job 기록=PII·쓰기 많음 → 다른 DB(예: Postgres). PII를 이 저장소 SQLite에 섞지 않는다.

**책임 배치**

| 관심사 | 앱 서버(BFF) | 추론 서버(이 저장소) |
|---|:---:|:---:|
| 인증·토큰·리프레시 | ✅ | ✕ |
| Job 큐·상태·폴링 | ✅ | ✕(동기 반환) |
| 유저·작업 기록 | ✅ | ✕ |
| `/v1`·오류봉투·레이트리밋 | ✅ | ✕ |
| VLM·검출·포즈·검색 | ✕ | ✅ |
| 포즈 라이브러리·BVH 제공 | ✕ | ✅ |

**안 고른 것 / 언제 뒤집히나** — **FastAPI 단일 처리**는 (a) 백엔드가 도원 1명뿐이고 별도 앱 서버 개발자가 없으며 (b) 사용자 수십 명 데모 규모이고 (c) 마감이 며칠일 때는 더 빠르다(인증 라우터 + `jobs` 테이블만 추가). 대가는 나중의 분리 리팩터링인데, **`/analyze` 인터페이스만 안 건드리면** 그 비용은 감당 가능하다.

**승격 경로(지금 할 것) — 아키텍처는 분리로 정하되 순서는 미룬다**
1. 이번 주: 클라는 Mock 어댑터 유지. 실연결 검증이 필요하면 추론 서버 `/analyze`를 **직접** 호출해 후보 뷰어까지만 확인.
2. 실제 인증이 필요해지는 시점에 **얇은 BFF 도입** — `/v1`·토큰·Job 폴링·오류봉투가 여기로 들어가고, 추론 서버는 손대지 않는다.

**구현 방향**: BFF는 **Node.js/Hono(TS)로 결정**(빌더가 클라 팀 → Tauri 클라와 타입 공유·같은 툴체인). 레포 `04JUNG/Standin-app-server`에 스캐폴드 완료. 스택·엔드포인트 매핑·Job 래핑·인증·모듈 구조·단계별 계획은 **`docs/BFF_DESIGN.md`** 참조.

**레포·소유(권장)**: **`04JUNG/Standin-app-server` 신설**(다른 두 레포와 같은 소유 아래). 추론 레포(`Standin-server`) 안에 넣지 않는다 — 도원의 순수 추론 경계 유지 + 빌더≠도원이라 소유·머지 충돌 회피. 만드는 사람(당신 또는 팀원)이 그대로 소유·CI·배포. **레포 생성은 Phase 0 착수 시점**(실인증 필요 시)에 하고, 그때 최소 스캐폴드부터 시작.

**⚠ 확인 필요(팀)**
1. BFF↔추론 서버 사이 내부 인증(서비스 토큰/네트워크 격리) 둘지 — 추론 서버는 무인증이라 공개 노출 금지.
2. Job: BFF가 동기 `/analyze`를 감싸 폴링을 제공(권장). 추론이 길어지면 추론 서버에도 Job 도입 재검토.

---

## 요약 (한 줄씩)
- **DB = SQLite**(단일 파일, 태그=컬럼), 스케일 시 `repo.py`만 pgvector/LanceDB로 교체.
- **저장 = 2D 키포인트 34차원 BLOB + 뷰별 미리 투영, BVH는 파일 경로**(확정). feature_version으로 정규화 정합성 보증.
- **태깅 = 라이브러리 action/shot은 VLM 반자동+검수(쿼리와 동일 어휘), view는 투영각에서 확정, relationship은 등록 시 지정.**
- **핸드오프 = inference가 base BVH 제공, BFF가 final base/refined bytes 확정, Converter가 V3.2 FBX·mirror 담당.** 라이브러리는 단일 소스.
- **계층 = 얇은 앱 서버(BFF) 분리**(인증·Job·기록·`/v1`은 BFF, 추론 서버는 순수 추론 유지). 단일 처리는 소규모·단기 MVP 한정. `/analyze` 인터페이스는 불변.
