# 포즈 라이브러리 확장 운영 가이드

> 상태: 기준
> 갱신일: 2026-08-18
> 기준 코드: `scripts/init_bvh_tag_inventory.py`, `scripts/build_semantic_tagging.py`,
> `scripts/validate_semantic_tagging.py`, `scripts/build_semantic_action_mapping.py`,
> `scripts/build_semantic_documents.py`, `scripts/provision_semantic_encoder.py`,
> `scripts/build_semantic_index.py`, `scripts/validate_semantic_index.py`, `src/posecode.py`,
> `src/semantic_catalog.py`, `src/semantic_documents.py`, `src/semantic_embedding.py`,
> `src/semantic_index.py`, `src/semantic_search.py`, `scripts/semantic_search.py`,
> `scripts/eval_semantic_search.py`,
> `scripts/build_db.py`, `scripts/mixamo_fbx_to_bvh.py`, `scripts/bvh_contact_sheet.py`,
> `scripts/eval_search.py`, `src/library.py`, `src/repo.py`
>
> 목적: 포즈 수를 무작정 늘리는 것이 아니라, **작가가 어려워하는 액션 컷의 Top-5 유효 후보율을
> 회귀 없이 높이는 것**을 목표로 한다. 라이선스 판정은 운영 가이드이며 법률 자문은 아니다.

---

## 1. 먼저 결론

확장 순서는 아래로 고정한다.

1. **기존 1,308개 포즈의 결정적 inventory를 만든 뒤 출처·라이선스·원본 clip ID를 복구한다.**
2. **BVH typed atom·파일명/source seed·검수된 caption을 승인한 뒤 다국어 dense 의미 색인을 만든다.**
3. 이미 가진 **CMU 스포츠 원본**에서 필요한 종목만 소량 추출한다.
4. 실제 러프에서 계속 비는 액션만 **Mixamo**에서 골라 보충한다.
5. 신규 오픈소스는 **ACCAD의 바로 읽히는 BVH**부터 5~10개로 호환성 시험한다.
6. **AIST++·AI Hub는 별도 실험 트랙**으로 둔다. 변환과 이용조건이 확정되기 전에는 제품
   라이브러리에 넣지 않는다.
7. **mocapdata.com 계열은 원문 라이선스를 확보하기 전까지 사용 보류**한다.

포즈 개수는 성공 지표가 아니다. 다음 질문에 `예`라고 답할 수 있는 배치만 승격한다.

- 고정 평가 러프에서 이전보다 "고쳐 쓸 만한 후보"가 늘었는가?
- 기존에 잘 나오던 컷의 Top-5가 나빠지지 않았는가?
- 한 원본 clip의 비슷한 프레임이 Top-5를 채우지 않는가?
- 모든 포즈의 출처·원본·변환·라이선스·배포 가능 범위를 추적할 수 있는가?
- 이 포즈를 고객에게 BVH로 전달해도 되는가?

---

## 2. 소스별 우선순위와 사용 판정

| 순위 | 소스 | 지금 할 일 | 제품 BVH 승격 판정 | 이유 |
|---:|---|---|---|---|
| 0 | 기존 라이브러리 | provenance 복구 | 기록 완료 전 금지 | 현재 `source=synthetic`, `license=n/a` 기본값이 섞여 있어 감사 불가 |
| 1 | CMU 스포츠 | 보유 데이터에서 복싱·농구·무술·달리기만 pilot | **조건부 검토** | 현재 파서가 CMU 계열 BVH 관절명을 지원하고 변환 공수가 가장 작음 |
| 2 | Mixamo 전투·스포츠 | 실제 빈 액션만 10~20 clip 단위로 보충 | **refine 후 승격 가능** | Standin에서는 원본이 아니라 러프에 맞춰 조정한 결과 BVH를 제품 산출물로 전달 |
| 3 | ACCAD Open Motion | 직접 제공 BVH 5~10개 호환성 시험 | **출처표기 후 유력** | CC BY 3.0이며 BVH 묶음이 있으나 관절명 호환성 확인 필요 |
| 4 | AIST++ | dance gap이 확인될 때만 별도 PoC | **검토 전 금지** | annotation은 CC BY 4.0이지만 SMPL→BVH 변환 경로와 관련 자산 라이선스 확인 필요 |
| 5 | AI Hub | 데이터 구조·이용목적 검증용 PoC만 | **금지/서면 확인 필요** | 현 정책은 AI 학습모델 학습용으로 용도를 제한하고 제3자 제공도 제한 |
| 6 | mocapdata.com | 원문 약관·배포 주체·파일별 license snapshot 확보 | **확인 전 금지** | 2010년 2차 기사만으로 정확한 CC 종류와 현재 권리관계를 확정할 수 없음 |

### 2.1 라이선스에서 반드시 나눠 볼 세 항목

`상업 가능` 한 칸으로 관리하지 않는다. 아래 셋을 별도로 판정한다.

1. `commercial_use`: 이 데이터를 상업 제품 제작에 사용할 수 있는가?
2. `raw_redistribution`: 원본 또는 변환 모션 파일을 제3자에게 줄 수 있는가?
3. `product_bvh_export`: Standin 고객에게 `/pose/{id}/bvh`로 전달할 수 있는가?

Standin은 이미지·영상만 만드는 제품이 아니라 **고객에게 BVH 파일을 전달**한다. 따라서
`commercial_use=yes`여도 `product_bvh_export=no`이면 제품 후보로 승격할 수 없다. Mixamo는 이
프로젝트의 결정에 따라 `delivery_mode=refined_only`일 때 제품 후보로 승격할 수 있다.

### 2.2 2026-08-13 확인 메모

- **CMU**: 공식 사이트는 상업 제품에서의 사용을 허용하고, 데이터를 직접 재판매해서는 안 된다고
  명시한다. FAQ에는 복사·수정·재배포 허용 문구도 있다. 두 문구와 Standin의 유료 BVH 전달 방식이
  정확히 어떻게 만나는지는 별도 검토가 필요하다. 저장소 정책은 계속 보수적으로 유지하여 공개
  Git·공개 버킷에 원본/변환 BVH를 넣지 않는다.
- **Mixamo**: Adobe FAQ는 상업 프로젝트 사용을 허용한다. Standin에서는 검색된 원본을 그대로
  전달하지 않고 러프에 맞춰 refine한 결과 BVH만 제품 산출물로 전달하는 것으로 결정했다. 따라서
  이 가이드에서는 Mixamo의 refine 결과 전달을 승격 차단 사유로 보지 않는다.
- **ACCAD**: 공식 페이지의 Open Motion Project는 CC BY 3.0이다. 저작자·출처·라이선스 링크와
  변경 사실을 제품 고지에 남기는 것을 승격 조건으로 둔다.
- **AI Hub**: 질문에 적힌 `dataSetSn=66`은 `스포츠 사람 동작(농구)` 영상 데이터다. 70종 동작과
  3D 관절 좌표가 있는 데이터는 `dataSetSn=209`의 `사람 인체/자세 3D`다. AI Hub 일반 정책은
  AI 데이터의 용도를 인공지능 학습모델 학습으로 한정하고, 승인 없는 제3자 제공을 금지한다.
- **AIST++**: Google이 제공한 annotation은 CC BY 4.0이다. 원본 영상·음악은 AIST Dance Video DB의
  별도 약관을 따른다. BVH 구축에는 annotation만 사용하고 영상·음악을 번들에 넣지 않는다. SMPL
  모델 또는 변환 도구를 쓸 경우 그 별도 라이선스까지 확인한다.
- **mocapdata.com**: `CC 계열`이라는 2차 기사만 저장하지 말고 정확한 원문 약관과 CC 버전을
  확보해야 한다. 원문을 확보하지 못하면 사용하지 않는다.

공식 확인 링크:

- [CMU Motion Capture Database](https://mocap.cs.cmu.edu/)
- [CMU FAQ](https://mocap.cs.cmu.edu/faqs.php)
- [Adobe Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html)
- [ACCAD MoCap System and Data](https://accad.osu.edu/research/motion-lab/mocap-system-and-data)
- [AI Hub 데이터 이용정책](https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)
- [AI Hub 스포츠 사람 동작(농구), dataset 66](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=66)
- [AI Hub 사람 인체/자세 3D, dataset 209](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=209)
- [AIST++ license와 데이터 설명](https://google.github.io/aistplusplus_dataset/factsfigures.html)
- [AIST++ download와 원본 영상 약관 경계](https://google.github.io/aistplusplus_dataset/download.html)

---

## 3. 확장 단위: “데이터셋 전체”가 아니라 “gap batch”

한 번에 한 데이터셋 전체를 넣지 않는다. 한 배치는 **하나의 검색 공백**을 해결한다.

권장 초기 backlog:

| 우선순위 | gap 묶음 | 예시 자세 |
|---:|---|---|
| P0 | 근접 전투 | 복싱 가드, 잽, 훅, 블록, 맞고 휘청임 |
| P0 | 무기 전투 | 검 들기, 찌르기, 베기, 막기, 창 던지기 준비 |
| P0 | 돌진·회피 | 정면 질주, 태클, 급정지, 몸 숙여 피하기, 구르기 전후 |
| P1 | 농구 | 드리블, 슛 준비·릴리스, 덩크, 리바운드, 수비 자세 |
| P1 | 무술·킥 | 앞차기, 옆차기, 낮은 자세, 착지, 균형 잃음 |
| P2 | 고동세 연출 | 점프 정점, 큰 몸통 비틀기, 춤의 실루엣이 큰 프레임 |

각 배치는 원본 clip 10~20개로 시작한다. clip당 대표 포즈는 기본 3개를 뽑고, 사람이 1~3개만
keeper로 남긴다. 첫 pilot의 최종 증분은 대략 20~50 pose면 충분하다.

새 gap의 우선순위는 다음 점수로 정한다.

```text
priority = 실제 실패 컷 수 × 작가 수정 난이도 × 재사용 가능성
```

`멋있어 보임`, `데이터가 많음`은 우선순위 근거로 쓰지 않는다.

---

## 4. 파일과 provenance 장부

모든 작업 파일은 `data/` 아래에 두며 Git에 커밋하지 않는다.

```text
data/library_work/<batch_id>/
├─ raw/             받은 원본. 수정 금지
├─ converted/       형식 변환 결과
├─ review/          대표 프레임 후보
├─ accepted/        사람 검수 통과 1-frame BVH
├─ rejected/        폐기본과 폐기 사유
├─ reports/         contact sheet·평가 결과·hash
├─ license/         약관 snapshot 또는 획득 당시 증빙
├─ manifest.v1.jsonl  포즈별 provenance 단일 소스
└─ manifest.csv       JSONL에서 만든 검수·교환용 파생 파일
```

배열, 필드별 근거, revision을 잃지 않도록 JSONL을 단일 소스로 쓴다. CSV는 사람이 검수하거나
외부 도구에 전달할 때만 매번 재생성하며, CSV 수정본을 그대로 색인 입력으로 쓰지 않는다.

`manifest.v1.jsonl`의 provenance 필수 필드(`manifest.csv`에도 평탄화해 포함):

| 필드 | 예시 | 규칙 |
|---|---|---|
| `pose_id` | `cmu_135_03_boxing_guard_f0120` | 전 라이브러리에서 유일하고 재빌드해도 바뀌지 않음 |
| `provider` | `cmu_graphics_lab` | 콘텐츠 제공처. rig·리타깃 프로필과 분리 |
| `source_clip_id` | `cmu:135_03` | Standin 내부 join key. 제공처 번호가 아님 |
| `native_subject_id` | `135` | 제공처가 쓰는 subject/session 번호 그대로 |
| `native_asset_id` | `null` | 제공처에 별도 asset/package 번호가 있을 때만 기록 |
| `native_clip_id` | `03` | 제공처가 쓰는 clip/take/trial 번호, leading zero 보존 |
| `original_title` | `Empi` | 공식 catalog 원문 이름 |
| `original_filename` | `135_03.amc` | 실제 받은 파일 basename; 확인 전에는 null |
| `pose_family_id` | `cmu_135_03` | 정규식 추측이 아니라 provenance로 부여 |
| `source_url` | 공식 상세 페이지 | 2차 블로그보다 원문 우선 |
| `acquired_at` | `2026-08-13` | 약관 변경 추적용 |
| `source_sha256` | 원본 hash | 원본 동일성 확인 |
| `bvh_sha256` | keeper BVH hash | 변환 산출물 동일성 확인 |
| `original_format` | `FBX` | 원본 형식 |
| `conversion_recipe` | `blender-4.x/mixamo-v1` | 도구·버전·옵션 포함 |
| `selected_frame_index` | `120` | 대표 프레임 재현용. `frame_index_base`와 함께 저장 |
| `sample_ordinal` | `2` | 두 번째 선택 pose라는 뜻. frame 번호와 분리 |
| `license_id` | `CC-BY-3.0` | 임의 요약 대신 정확한 이름 |
| `license_url` | 원문 | 링크 또는 계약 문서 식별자 |
| `commercial_use` | `yes` | `yes`, `no`, `review` |
| `raw_redistribution` | `no` | `yes`, `no`, `review` |
| `product_bvh_export` | `no` | **제품 승격 게이트** |
| `delivery_mode` | `refined_only` | `original`, `refined_only`, `internal_only` |
| `attribution_text` | 고지 문구 | 필요 없으면 `n/a` |
| `semantic_seed_labels` | `boxing;guard;defense` | 검수 후보 seed. 검색용 typed atom·document의 확정값이 아님 |
| `qa_status` | `accepted` | `pending`, `accepted`, `rejected` |
| `reviewer` | 검수자 ID | 누가 시각 확인했는지 기록 |
| `reject_reason` | `duplicate` | accepted면 비움 |

현재 `build_db.py`는 manifest를 읽지 않고 파일명 기반 임시 태그만 만든다. 따라서 지금 명령으로
만든 실 DB는 `source`·`license`가 올바르게 채워지지 않는다. **manifest→`LibraryEntry.meta` import와
`product_bvh_export` enforcement가 생기기 전 DB는 기하 평가용 candidate DB일 뿐, 출시 DB가 아니다.**

### 4.1 2026-08-13 기준 1,307개에서 비어 있던 것

2026-08-13 로컬 실측 기준이다. `data/bvh/_dupes/`의 격리 파일 74개는 아래 집계와 색인 대상에서
제외했다.

| 항목 | 현재 상태 | 태깅에 주는 의미 |
|---|---:|---|
| root BVH / projection | 1,307 / 5,228 | 4-view 기하 색인은 이미 존재 |
| 1-frame BVH | 1,307 | 시간 정보 없이 정적 자세만 관찰 가능 |
| 원본/미러 | 원본 653, `_mirror` 654 | 정상 쌍 653, orphan mirror 1개 |
| 의미 검수 단위 | 654 | 원본+미러의 방향 중립 caption을 한 번만 검수 |
| `action=other` | 1,275 (97.6%) | 현재 action 값은 의미 색인 재료로 부족 |
| `source=synthetic`, `license=n/a` | 각각 100% | 실제 출처가 아니라 import 기본값이므로 복구 필요 |
| `meta_json`, `set_id` | 전부 빈 값, 0개 | provenance와 다인 set은 아직 미구축 |

orphan은 `rokoko_Typing_UsingMouse_mixamo_00882_mirror`다. `_dupes/_removed_list.txt`와 실제 격리
파일도 일치하지 않으므로, 목록 이름만 믿지 말고 hash와 폐기 판정 버전을 새 원장에 기록한다.

### 4.2 ID 다섯 종류를 합치지 않는다

| ID | 단위 | 용도 |
|---|---|---|
| `pose_id` | 실제 BVH 1개 | 기하 색인·export·좌우 posecode의 기준 |
| `source_clip_id` | 원본 motion clip | provenance와 행동명 seed 상속 |
| `mirror_group_id` | 같은 정적 프레임의 원본+미러 | 좌우 변환 검증 |
| `semantic_unit_id` | 방향 중립 caption 검수 단위 또는 실제 다인 set | 의미 태그와 사람 결정의 기준 |
| `set_variant_id` | 한 interaction set의 완전한 original 또는 mirrored instance | role 원자 반환과 혼합 방지 |

기존 `pose_family_id`는 검증된 source provenance를 바탕으로 **검색 결과 다양성**에 쓰는 별도 키다.
이를 `_mirror` 접미사를 지운 값으로 자동 확정하지 않는다. 미러 두 포즈는 기하적으로 대부분 멀고
각각 정당한 후보이므로 기하 색인에서 하나로 접거나 삭제하면 안 된다.

같은 source clip에서 뽑은 다른 프레임도 각각 다른 `pose_id`와 `semantic_unit_id`를 갖는다. 행동명
seed만 상속할 수 있을 뿐 caption·posecode를 복사하지 않는다. `_ground`, `_legfix`, `_legstraight`,
프레임 번호도 `mirror_group_id`를 만들 때 제거하지 않는다. 현재의 `semantic_unit_id`는
`pose:<mirror_group_id의 base>` 형식이고, 실제 다인 세트만 `set:<set_id>`를 쓴다.

다인 set을 mirror할 때는 한 semantic unit 안에 `set_variant:<set_id>:original`과
`set_variant:<set_id>:mirrored`를 따로 두고 `mirror_of_set_variant_id`로 연결한다. 각 variant 안에는
expected role이 정확히 한 번씩 있어야 한다. original A와 mirrored B를 섞거나 두 variant의 전 멤버를
한 후보로 반환하면 build 실패다.

### 4.3 원본 출처·이름·번호 원장

원본 출처·이름·번호는 semantic proposal에 복사하지 않고 두 원장으로 정규화한다.

```text
source_clips.v1.jsonl   원본 clip 1개당 1행
  └─ provider, collection, native subject/asset/clip ID,
     official title, original filename/URL/hash, catalog·license evidence

pose_lineage.v1.jsonl   concrete BVH 1개당 1행
  └─ library_no, pose_id, source_clip_id, selected frame,
     sample ordinal, parent artifact, retarget·ground·fix·mirror operations
```

`이름`과 `번호`를 각각 한 칸으로 만들지 않는다.

| 사용자가 보는 개념 | 저장 필드 | 예시·주의 |
|---|---|---|
| 원본 제공처 | `provider` | CMU·Rokoko 등. `mixamorig` rig 이름은 제공처가 아님 |
| 공식 원동작 이름 | `original.title` | 제공처 catalog 원문. 검색용 번역·alias와 분리 |
| 실제 받은 파일명 | `original.filename` | 이름이 같아도 hash가 다를 수 있으므로 그대로 보존 |
| 제공처 원본 번호 | `native_ids.subject_id/asset_id/clip_id` | leading zero를 보존하는 문자열 |
| Standin 내부 번호 | `library_no` | `BVH-000001` 형식의 append-only 번호. 행 순서로 재계산 금지 |
| 원본 프레임 번호 | `selected_frame_index` | 반드시 `frame_index_base`와 함께 저장 |
| 선택 포즈 순번 | `sample_ordinal` | `Waving_02`의 `02` 같은 추출 결과 순번; frame과 다름 |
| 리타깃·보정 | `derivation.operations` | `retarget`, `ground`, `legfix`, `legstraight`, `mirror` |

`source_clip_id`는 두 원장의 join key이며 제공처가 발급한 번호가 아니다. 제공처 원본 번호는 반드시
`native_ids.*`에 원문 그대로 둔다. 기존의 모호한 `asset_id`, `clip_id`, `frame`, generic `number` 한
칸으로 합치지 않는다.

현재 파일명의 숫자는 세 경우로 나뉜다.

- `cmu_05_03_00150`: `05=subject`, `03=trial`, `00150=선택 frame hint`
- `Big Side Hit_00018`: 로컬 원본과 비교해 0-based frame 18임을 실제로 검증 가능
- `Waving_02`: `02=두 번째 선택 샘플`; 원본 frame은 conversion log 없이는 복구 불가

`rokoko_*_mixamo_00162`도 `Rokoko=provider hint`, `mixamo=retarget/rig hint`, `00162=frame hint`로
분리한다. 원본 catalog나 변환 log가 없으면 이 값들을 verified 필드로 승격하지 않는다.

CMU처럼 공식 catalog가 있는 source는 subject/trial을 join해 `original.title`을 복구할 수 있다.
예를 들어 `cmu_05_03`은 [CMU 공식 catalog](https://mocap.cs.cmu.edu/search.php?subjectnumber=5&trinum=3)에서
subject 5, trial 3,
`dance - sideways arabesque, turn step, folding arms`로 확인된다. 그러나 로컬 원본 AMC/BVH 또는 변환
log가 없으면 `00150`의 frame base와 실제 계보는 계속 `unverified`다.

실제 형식과 검증 수준이 다른 샘플은
[`templates/bvh_tagging/provenance_samples.v1.jsonl`](templates/bvh_tagging/provenance_samples.v1.jsonl)에
있다. `source_clip`은 catalog 단위로 한 번 검수하고 여러 BVH가 참조하므로 같은 원본명을 1,307행에
반복 입력하지 않는다.

### 4.4 의미 태깅 원장과 템플릿

운영 산출물은 Git에서 제외된 `data/semantic/`에 둔다.

```text
data/semantic/
├─ inventory.v1.jsonl       # concrete BVH별 경로·hash·QA·미러 관계
├─ source_clips.v1.jsonl     # 원본 clip별 출처·이름·제공처 번호
├─ pose_lineage.v1.jsonl     # concrete BVH별 내부 번호·frame·파생 이력
├─ library_numbers.v1.json   # 삭제·재추가에도 재사용하지 않는 append-only 내부 번호 registry
├─ proposals.v1.jsonl       # posecode·파일명·VLM 자동 제안, append-only
├─ decisions.v1.jsonl       # 사람 승인·수정·거절, append-only
├─ review_queue.csv         # 위 JSONL에서 재생성하는 검수 화면용 파일
├─ provenance_review_queue.csv # 출처·이름·번호·frame 검수용 파생 파일
├─ tagging_review.v1.db     # 자동 proposal/atom을 조회하는 비제품 작업 색인
├─ tagging-summary.v1.json  # 배치 집계와 production blocker
├─ tagging-validation.v1.json # 구조 검증 결과와 남은 review item
├─ action_mapping.v2.jsonl  # source 이름→canonical context + fallback 검색 coverage
├─ action_mapping_review.v2.csv # canonical mapping과 선택적 사람 검수 뷰
├─ action-mapping-summary.v2.json # mapping·검색 coverage 집계
├─ search_documents.v2.jsonl # active unit별 최종 observed/context 문서 세트
├─ search-document-summary.v2.json # 문서 coverage·버전·fingerprint 집계
├─ snapshots/               # 공식 catalog 1회 snapshot과 hash 근거
└─ builds/<semantic_build_id>/
   ├─ semantic-build.json  # 승인 revision·passage·embedding 설정 고정
   └─ pose_semantics.db    # staging 산출물; runtime DB는 SEMANTIC_DB_PATH로 활성 bundle을 가리킴
```

버전 관리되는 형식 예시는 [`templates/bvh_tagging/`](templates/bvh_tagging/)에 있다.

- `pose_member_inventory`: 실제 BVH 1개의 무결성·출처·미러·set·색인 가능 상태
- `semantic_proposal`: 방향 중립 caption·unit 의미 atom과 멤버별 좌우 posecode/observed atom 제안
- `review_decision`: `accepted | accepted_with_edits | rejected | blocked`와 수정 이력
- `review_queue.csv`: 편집하기 쉬운 파생 뷰이며 원본 기록이 아님

decision은 proposal 전체가 아니라 `reviewed_fields`에 적힌 필드만 승인한다. 승인 목록 밖의 자동
제안은 renderer가 폐기하거나 `unknown`으로 남긴다. reviewed field 안의 atom/posecode가 proposal에서
`generated | needs_review`였더라도 유효한 decision overlay가 그 필드의 최종 승인 상태를 제공한다.
resolver는 이 규칙을 적용한 effective revision을 만들며 원 proposal을 덮어쓰지 않는다.

필드 승인이 부족하다는 이유만으로 unit 전체를 검색에서 제거하지 않는다. 구조 검증을 통과한 BVH
posecode는 `observed` 채널로 자동 색인하고, 원본 이름은 `contextual/candidate_only` 채널로만 색인한다.
미승인 source 이름과 canonical 후보는 pose truth 또는 hard filter가 될 수 없다.

`semantic_index`에 들어가는 accepted unit의 최소 요건은 다음뿐이다. JSONL은 원장이고
`pose_semantics.db`와 embedding은 승인 원장에서 재생성하는 build artifact다.

- 방향 중립 `caption_ko` 1개. 다국어 dense는 한국어를 직접 처리하므로 `caption_en`은 필수가 아님
- concrete 멤버 전부의 버전된 posecode, 연속 측정값과 typed observed atom
- `coarse_action`, posture/gesture, alias·소품·style이 있으면 typed intended/contextual atom과 근거
- intended/contextual hard facet의 필드별 provenance와 사람 decision. observed posecode와
  candidate-only source context에는 사람 decision을 필수로 요구하지 않음
- 좌우 쿼리를 처리할 수 있는 member별 posecode와 완전한 원본/미러 연결

사람이 embedding용 문장을 여러 개 따로 쓰지 않는다. builder가 승인 caption·alias·source context와
typed atom을 버전된 template로 짧은 text document에 결정적으로 렌더링한다. embedding은 정확한
model/revision/tokenizer/prefix/pooling/차원/정규화 설정으로 오프라인 생성한다.

fine action, motion phase, prop, interaction은 억지로 채우지 않는다. 관찰·source·사람 근거가 없으면
빈 배열 또는 `unknown`이 올바른 태그다. 이 빈 값은 검색 제외 조건이 아니며 posecode·raw source
context·안전하게 판정된 다른 facet 채널은 계속 사용한다.

annotation이 `semantic_index`에 들어갈 수 있다는 것과 제품에서 BVH를 선택·전달할 수 있다는 것은
다른 gate다. 초기 semantic selectable pool은 `product_bvh_export=yes && delivery_mode=original`만
허용한다. `refined_only`는 기존 geometry→refine 제품 경로에서는 사용할 수 있지만 의미 후보에는
refine을 금지하므로 semantic Top-K에서 제외한다. 개발 진단에 표시할 때도 `selectable=false`이고
원본 BVH URL을 주지 않는다.

초기 inventory는 다음처럼 만든다.

```bash
python scripts/init_bvh_tag_inventory.py \
  --bvh-dir data/bvh \
  --output data/semantic/inventory.v1.jsonl
```

스크립트는 의미·출처·라이선스를 지어내지 않는다. root-level BVH만 스캔하고 SHA-256, frame/joint 수,
COCO-17 body mapping, 미러 쌍, 보수적인 파일명 힌트를 기록한다. 현재 데이터에서 기대하는 요약은
`1,308 poses / 654 semantic units / 654 mirror pairs / orphan 0`이다.

inventory 이후의 provenance·posecode·자동 의미 제안·검수 색인은 다음처럼 만든다. CMU HTML은 공식
catalog 전체 페이지를 한 번 저장한 snapshot이며, snapshot hash와 획득일을 원장에 기록한다.

```bash
python scripts/build_semantic_tagging.py \
  --inventory data/semantic/inventory.v1.jsonl \
  --bvh-dir data/bvh \
  --raw-dir data/_action_raw \
  --output-dir data/semantic \
  --cmu-catalog-html data/semantic/snapshots/cmu-search-20260814.html \
  --cmu-catalog-captured-at 2026-08-14

python scripts/validate_semantic_tagging.py \
  --output-dir data/semantic \
  --report data/semantic/tagging-validation.v1.json

python scripts/build_semantic_action_mapping.py
python scripts/build_semantic_documents.py
pip install -r requirements-semantic.txt
python scripts/provision_semantic_encoder.py
python scripts/build_semantic_index.py
python scripts/validate_semantic_index.py \
  --build-dir data/semantic/builds/217d56e31c42634ec920db8704dd151b088b2f12291d5e3726f5b34c9be50196
python scripts/build_golden_queries_v2.py
python scripts/eval_semantic_search.py --split development
```

마지막 두 builder는 source 수를 328로 고정하지 않는다. 새 라이브러리가 들어오면 활성 source 전체를
다시 매핑한 뒤 unit별 최종 문서를 만든다. canonical 행동을 만들지 못한 source도 posecode semantic
unit이 연결되지 않으면 빌드를 실패시켜 조용한 검색 공백을 막는다.

`tagging_review.v1.db`는 자동 제안과 관찰 atom을 검색·검수하기 위한 **비제품 작업 색인**이다.
여기에는 embedding이 없으므로 새로 생성한 staging `pose_semantics.db`와 혼용하지 않는다. staging
DB는 내부 runtime/API development 평가를 통과했지만 holdout/release bundle 승격 전에는
`production_ready=false`다.
`library_numbers.v1.json`은 다음 150개가 들어올 때 기존 `BVH-000001`~번호를 보존하고 새 pose에만
다음 번호를 배정한다. active 행 순서로 번호를 다시 계산하거나 삭제된 번호를 재사용하지 않는다.

추가 BVH가 들어오면 기존 `data/semantic/` 원장을 지우지 않고 위 세 명령을 같은 output 경로로 다시
실행한다. inventory와 review DB·CSV는 active 전체에서 재생성하고, 기존 `library_no`는 registry에서
보존하며, 동일 `run_key` proposal은 중복 추가하지 않는다. BVH hash나 posecode 규칙이 달라진 unit만
새 proposal revision을 append한다. 그다음 action mapping builder를 실행해 source별 fallback coverage를
다시 검증한다. 기존 decision을 새 revision에 자동 승계하지 않고 다시 검수한다.

### 4.4.1 2026-08-17 현재 배치 결과

> **2026-08-18 후속 상태:** `rokoko_Typing_UsingMouse_mixamo_00882` 원본 짝 생성 후
> 1,308 pose / 654 semantic unit / 654 complete mirror pairs / orphan 0이다. 행동명이 없던 CMU
> 35 source clip(38 unit)은 `config/library_exclusions.v1.json`에서 PX 검색 제외·실제 삭제 대기로
> 전환해 P0/P1은 0이다. 이름 있는 활성 328 source clip은 v2로 매핑했고, 행동 ID가 비어 있는
> 46 source clip/94 unit도 posecode·source context로 검색 가능하다. 표준화 신규 작업은
> `config/semantic_vocab.v2.json`을 따른다.

활성 `data/bvh`와 `data/poses.db`의 1,308개를 대상으로 실행했다. 별도
`data/pose-library-v1/bvh`에만 있는 58개는 활성 geometry DB 밖이므로 이번 원장에 섞지 않았다.

| 항목 | 결과 |
|---|---:|
| BVH parse / COCO body mapping 실패 | 0 / 0 |
| pose member / semantic unit | 1,308 / 654 |
| source clip 원장 | 363 |
| CMU source clip / catalog record 연결 / 공식 title 복구 | 253 / 232 / 218 |
| 로컬 원본 BVH source clip / exact frame lineage | 25 / 51 poses |
| 결정적 member / 방향 중립 unit observed atom | 14,409 / 5,405 |
| mirror atom 직접 통과 / 원본 기준 자동 정규화 / 검수 필요 | 593 / 61 / 0 pairs |
| 검수 우선순위 | P2 자동 검증 616 / PX 검색 제외 38 units |
| 행동명 누락 | 35 source clips / 38 units / 76 BVHs |
| 최종 검색 문서 세트 / text document / dense embedding | 616 / 2,892 / 2,892 |

구조 validator는 오류 0건으로 통과했다. CMU 항목은 catalog 상세 연결 여부와 무관하게 출처를
`CMU`로 표시하고, 공통 출처·라이선스 확인을 태깅 우선순위에서 제외한다. Mirror bucket이 달랐던
61 unit은 원본 관찰 atom을 좌우 중립화한 검색 태그로 자동 정규화했으며, 양쪽의 원 측정값은
진단용으로 보존했다. 행동명 없는 38 unit은 PX로 제외했고 orphan mirror는 복구해 태깅 검수
대기 항목은 0개다. 활성 616 unit은 최종 검색 문서와 pinned E5 staging DB 생성까지 완료했다.

골든 쿼리용 atom 진단에서는 `왼쪽 다리를 뒤로 들고 양팔은 활짝 편 자세`의 필수 관절 조건을 모두
만족하는 concrete member 3개를 찾았다. `옛 전통 춤을 추는 자세`는 dance source context unit 36개가
있지만 승인된 `traditional`·`historical` atom은 0개이므로 현재 exact 판정은 의도대로
`library_gap`이다. 이는 dense 순위 평가가 아니라 자동 태깅 coverage의 구조 진단이다.

### 4.5 필드별 태깅 책임

| 정보 | 생성 주체 | 확정 가능한 것 | 확정하면 안 되는 것 |
|---|---|---|---|
| BVH 규칙 | 결정적 로컬 코드 | 관절 각도·상대 높이·근접·상체 기울기·stance의 `observed` atom | 행동, 감정, 소품, 다인 관계 |
| 파일명·source catalog | parser + 원본 원장 | clip 이름, frame, action/prop `intended` **hint**, provenance | 선택 프레임의 순간 행동을 정답으로 확정 |
| 4-view VLM | 오프라인 batch | 보이는 정적 자세의 방향 중립 caption·facet 제안 | 좌우 판정, 보이지 않는 소품, 관계, 자동 승인 |
| 사람 | 우선순위 검수 | caption·alias·행동 의도·문화/style context·소품 근거·실제 set | 잘못된 posecode 숫자를 임의 덮어쓰기 |

BVH posecode는 연속 측정값과 검색용 bucket을 함께 저장한다. 임계값이 바뀌어도 원 측정값으로
재계산할 수 있어야 한다.

- 좌우 팔꿈치·무릎 굽힘 각도
- 손목이 hip·torso·shoulder·head보다 위/아래인지
- 팔·다리의 body-local 방향
- 몸통의 전후·좌우 기울기
- 양발 간격과 서기·앉기·무릎 꿇기 후보
- 손-손, 손-머리, 손-몸통, 손-엉덩이, 손-허벅지 근접

`waving`, `throwing`, `attacking`, 감정, 칼·창·휴대폰, 손가락·손바닥 방향은 이 규칙으로 만들지
않는다. 현재 좌표는 hips 중심이므로 발 높이만 보고 `airborne`이나 실제 지면 접촉을 확정하지도
않는다. `motion_phase`는 단일 프레임에서 판정하기 어려우므로 source 근거나 사람 검수가 없으면
`unknown`이다.

### 4.6 실제 태깅 순서

```text
inventory
→ provenance_verified
→ posecode_measurements_and_typed_atoms_generated
→ filename/source_seeded
→ canonical_source_mapping + fallback_search_coverage
→ VLM_proposed(필요 항목만)
→ auto_verified_observed_tags | needs_review(P0/P1)
→ accepted | accepted_with_edits | rejected | blocked
→ approved text documents rendered
→ embeddings generated
→ 별도 pose_semantics.db validated
```

1. **Inventory:** 경로·hash·파싱·COCO mapping·1-frame·미러 관계를 전수 검사한다.
2. **Provenance:** 원본 manifest/catalog와 join해 source asset/clip/frame과 라이선스를 복구한다.
   파일명 정규식 결과는 `*_hint`일 뿐 verified 값이 아니다.
3. **Posecode/atom:** 1,308개 concrete BVH 각각에서 연속 측정값과 typed observed atom을 결정적으로
   계산한다. 원본과 미러도 각각 계산한 뒤 좌우 swap+반사 일관성을 자동 검사한다. 임계값 차이가
   생긴 쌍은 원본 atom을 좌우 중립화해 unit 검색 태그로 사용한다.
4. **Seed:** source 이름과 사람이 읽을 수 있는 파일명을 action·alias·intended prop 후보로만 쓴다.
5. **Canonical mapping/fallback:** 이름은 vocabulary v2로 결정적으로 매핑하되 불명확한 행동 ID는
   비운다. 모든 active source가 observed posecode로 검색 가능한지 검사하고, 원본 이름은
   candidate-only 문맥으로 유지한다.
6. **4-view VLM:** canonical 원본을 front/three-quarter/side/back으로 한 번 렌더한다. 첫 입력에서는
   파일명을 숨겨 보이는 정적 자세만 strict JSON으로 받고, 파일명 근거는 나중에 별도 합친다.
7. **사람 검수:** caption·alias·행동 문맥·소품 근거·충돌만 확인한다. 같은 mirror group의 방향 중립
   의미는 한 번 검수하고, 좌우 posecode 변환은 별도로 확인한다.
8. **Build:** validator가 승인 revision에서 text document를 렌더링하고 pinned multilingual E5로
   embedding한 뒤 staging `data/semantic/builds/<semantic_build_id>/pose_semantics.db`를 전체 빌드한다.
   개발 기본 runtime은 `SEMANTIC_DB_PATH=data/pose_semantics.db`, release는 활성 version bundle의
   절대 경로다. 기존 기하 `data/poses.db`에 semantic row를 덧대지 않는다.

VLM은 654개 unit 전부에 무조건 호출하지 않는다. 먼저 CMU처럼 이름이 불투명한 unit, 파일명과
posecode가 충돌하는 unit, 개발 검색에서 반복 실패한 unit을 처리한다. model ID, provider,
prompt version, contact-sheet hash를 반드시 저장한다. VLM의 자기 confidence는 승인 근거가 아니다.

### 4.7 처리·사람 검수 우선순위

| 우선순위 | 전수 처리 대상 |
|---|---|
| P0 | 파싱/mapping 실패, 중복 hash, orphan mirror, 제공처 자체가 불명확한 항목, 불완전 set |
| P1 | 파일명-VLM-posecode 충돌, 소품·상호작용 hard facet, 반복 검색 실패 |
| P2 | posecode 생성 성공, 원본 기준 mirror 정규화 성공 — 행동명 없음/unknown도 observed-only 자동 검색 |

P0/P1만 전수 처리한다. CMU는 모든 행에 `CMU`를 반복 표시하고 공통 라이선스 검토를 태깅 큐에서
제외한다. P2의 결정적 observed tag는 별도 사람 목록을 만들지 않고 넘어간다.
`rejected`는 자동 제안이 틀렸다는 뜻이며 BVH 삭제가 아니다. 실제 포즈 제외는 별도
`asset_status=excluded`로 남긴다. 사람은 proposal을 직접 덮어쓰지 않고 decision의 `edits`에
수정 내용을 기록한다.

### 4.8 색인·재실행 게이트

색인 가능 상태를 하나의 boolean으로 뭉치지 않는다.

- `geometry_engineering`: parse·body mapping·1-frame 검사를 통과하면 내부 기하 평가에 사용 가능
- `semantic_index`: semantic proposal이 사람에게 승인되고 vocab/version 검사를 통과해야 가능
- `release`: source·license·고객 BVH 전달 정책까지 검증해야 가능

같은 입력의 자동 태깅을 중복 생성하지 않도록 아래 값을 hash한 `run_key`를 쓴다.

```text
input_fingerprint(member BVH hash + contact-sheet hash)
+ semantic_schema_version
+ semantic_vocab_version
+ filename_parser_version
+ posecode_version + coordinate_profile
+ prompt_version + exact model_id
```

이 `run_key`는 proposal 재현용이며 embedding 설정을 섞지 않는다. 의미 build는 별도로 다음을 hash한
`semantic_build_id`를 쓴다.

```text
accepted proposal/decision revision set + pose_library_version
+ geometry_manifest_sha256 + geometry_db_sha256
+ semantic/atom/posecode version + passage_template_version
+ embedding_version(model/revision/tokenizer/prefix/pooling/dimension/normalization)
+ encoder_artifact_sha256
+ query_parser_version + resolution_policy_version + retrieval_policy_version
+ retrieval config(document aggregation/type cap + dense/lexical depth + FTS config + RRF k + constraint order)
```

규칙이나 모델이 바뀌면 기존 승인 기록을 덮지 않고 새 proposal revision을 만든다. build는 다음을
모두 통과하지 못하면 fail-closed한다.

- 모든 active `pose_id`가 정확히 한 inventory 행과 semantic unit에 속함
- path/hash 중복, orphan member, 불완전 set 없음
- `set_variant_id`별 expected role이 유일·완전하고 mirrored set 연결이 일관됨
- mirror의 방향 중립 의미는 같고 좌우 posecode는 올바르게 교환됨
- enum 밖 값, 빈 caption, 중복 alias, caption-atom 모순 없음
- prop에는 `source_catalog | filename | human` 중 하나 이상의 명시적 근거가 있음
- 한 build 안의 schema/vocab/posecode/prompt version이 일치함
- `semantic_build_id`가 `pose_library_version`과 실제 accepted proposal/decision revision을 고정함
- accepted unit마다 승인 text document와 embedding이 하나 이상 존재함
- embedding은 finite이고 지정 dimension·L2 norm·document hash·`embedding_version`이 일치함
- 실제 `pose_id → BVH hash`, geometry DB hash, encoder artifact hash가 build metadata와 일치함
- `observed | intended | contextual` provenance가 보존되고 미확인을 부재로 변환하지 않음

development 질의 결과를 보고 검수 우선순위를 올릴 수는 있지만, holdout 질의나 결과를 caption·alias
수정에 사용하지 않는다. holdout을 열고 수정했다면 해당 질의는 development로 강등하고 새 holdout을
마련한다.

---

## 5. 소스별 실제 작업법

### 5.1 CMU 스포츠 — 가장 먼저

1. 보유 중인 원본 목록에서 스포츠 subject/trial만 추린다.
2. 이미 변환해 둔 CMU/cgspeed 계열 BVH가 있으면 그것부터 사용한다.
3. 원본이 ASF/AMC뿐이면 변환 recipe와 툴 버전을 먼저 고정한 뒤 소량만 변환한다.
4. 복싱·농구·무술·달리기를 각각 별도 batch로 관리한다.
5. 원본 subject/trial ID를 `native_subject_id`·`native_clip_id`에 각각 보존한다.

주의: `build_db.py`는 멀티프레임 BVH의 **0번 프레임만** 읽는다. 원본 멀티프레임 BVH를 그대로
넣지 말고, 대표 프레임을 고른 1-frame BVH로 만든 뒤 `accepted/`에 넣는다. 현재 저장소에는
일반 BVH용 대표 프레임 추출기가 없으므로, 수동 pilot 후 자동화 작업을 별도로 잡는다.

### 5.2 Mixamo 전투·스포츠 — gap만 보충

1. gap backlog에 이름이 있는 clip만 받는다. 캐릭터 mesh는 라이브러리 목적에 필요 없다.
2. 한 번에 10~20 clip을 넘기지 않는다.
3. 기존 Blender 스크립트로 clip당 대표 포즈 3개를 뽑는다.

```bash
blender --background --python scripts/mixamo_fbx_to_bvh.py -- \
  data/library_work/<batch_id>/raw \
  data/library_work/<batch_id>/review \
  --num 3
```

4. 같은 clip의 포즈는 같은 `source_clip_id`·`pose_family_id`를 쓴다.
5. Standin 제품용 keeper는 `product_bvh_export=yes`, `delivery_mode=refined_only`로 기록한다.
   원본·미조정 BVH는 내부 검색과 refine 입력으로만 사용한다.

### 5.3 ACCAD — direct BVH부터

1. `Female 1`, `Male 1`, `Male 2`처럼 공식 페이지가 직접 제공하는 BVH 묶음에서 pilot을 고른다.
2. C3D만 제공되는 martial arts subset은 후순위로 둔다. C3D→rig→BVH는 별도 변환 프로젝트다.
3. ACCAD 관절명이 현재 `src/bvh.py`의 CMU/Mixamo suffix와 다를 수 있으므로 5~10개를 먼저
   파싱한다.
4. mapping을 추가해야 한다면 `src/bvh.py` 한쪽만 임시 수정하지 않는다. 색인과 refine이 공유하는
   매핑이므로 테스트와 4-view audit를 함께 갱신한다.
5. attribution 문구와 `CC-BY-3.0`, 원본 URL, 변경 사실을 manifest에 넣는다.

### 5.4 AIST++ — dance gap이 확인된 뒤

1. 영상·음악이 아니라 CC BY 4.0 annotation만 작업 범위로 삼는다.
2. 공식 `ignore_list.txt`에 있는 품질 불량 sequence는 제외한다.
3. SMPL pose/root trajectory에서 canonical skeleton의 1-frame BVH로 가는 재현 가능한 converter를
   먼저 만든다.
4. converter가 SMPL-Model 파일·코드에 의존하면 해당 라이선스를 별도로 판정한다.
5. 변환 10개에 대한 축·좌우·관절 회전 audit와 상업 사용 검토가 끝나기 전에는 제품 DB에 넣지
   않는다.

### 5.5 AI Hub — 별도 실험 트랙

- 농구 영상이 필요하면 dataset 66, 70종 3D 관절이 필요하면 dataset 209를 구분한다.
- dataset 209는 3D 좌표와 회전 JSON이지 바로 쓸 수 있는 BVH가 아니다.
- 현재 정책상 직접 검색 라이브러리·고객 BVH 전달에 사용하는 것은 허용된다고 가정하지 않는다.
- 사용 목적을 설명해 AI Hub/NIA 또는 수행기관의 **서면 확인**을 받은 뒤 진행한다.
- 승인되더라도 관절→계층·회전 복원→BVH 리타게팅 정확도를 별도 검증한다.

---

## 6. 배치별 QA 절차

### Gate 1 — 파싱과 관절 매핑

모든 keeper는 다음을 만족해야 한다.

- BVH 파싱 성공
- COCO-17 body 관절 12개 매핑 성공
- NaN/Inf 없음
- 좌우 팔·다리가 뒤바뀌지 않음
- root·축·단위가 한 source 안에서 일관됨
- 한 프레임짜리 BVH이며 선택 프레임을 manifest에서 재현 가능

`build_entries_from_pose()`는 body 관절이 빠지면 실패하므로 이 검사를 우회하지 않는다.

### Gate 2 — 4-view 눈검수

먼저 source당 8개를 4방향으로 본다.

```bash
python scripts/bvh_contact_sheet.py \
  data/library_work/<batch_id>/accepted \
  data/library_work/<batch_id>/reports/audit-4view.png \
  --views front,three_quarter,side,back --limit 8
```

그 다음 전체 front grid를 본다.

```bash
python scripts/bvh_contact_sheet.py \
  data/library_work/<batch_id>/accepted \
  data/library_work/<batch_id>/reports/keepers-front.png \
  --view front --cols 8
```

눈검수에서 버릴 것:

- 준비 자세만 있고 액션 정점이 없는 프레임
- 관절 꺾임·발 미끄러짐·root 폭주가 보이는 포즈
- 실루엣이 기존 포즈와 사실상 같은 포즈
- 같은 clip에서 차이가 거의 없는 프레임
- 칼·공 같은 소품이 없으면 의미가 사라지는 포즈인데 자세만으로도 구분된다고 오표기한 항목

### Gate 3 — candidate DB

기존 `data/poses.db`를 직접 덮어쓰지 않는다. `accepted/`의 BVH를 평평한 한 폴더에 모은 뒤 별도
경로로 빌드한다. 현재 스캐너는 하위 폴더를 재귀 탐색하지 않는다.

```bash
DB_PATH=data/poses.candidate.db \
BVH_DIR=data/library_work/<batch_id>/accepted \
python scripts/build_db.py
```

주의: 이 명령은 지정 DB의 기존 행을 전부 교체한다. 항상 candidate 경로를 사용하고, 여러 source를
합칠 때는 승인된 BVH를 하나의 staging 폴더에 모아 전체 DB를 다시 만든다.

### Gate 4 — 자기 쿼리 sanity check

배치에서 5개 이상을 골라 자기 자신이 `#1`, `dist=0.000`으로 돌아오는지 확인한다.

```bash
python scripts/eval_search.py \
  --from-bvh data/library_work/<batch_id>/accepted/<pose_id>.bvh \
  --db data/poses.candidate.db --topk 5 \
  --out data/library_work/<batch_id>/reports/<pose_id>-self.png
```

실패하면 라이브러리를 늘리지 말고 축·투영·피처 대칭 문제부터 고친다.

### Gate 5 — 고정 러프 전/후 비교

같은 러프, 같은 RTMPose/VLM cache, 같은 설정으로 baseline DB와 candidate DB를 비교한다.

사람이 인물별 Top-5에 대해 아래 세 값 중 하나를 표시한다.

- `useful`: 바로 쓰거나 조금 고쳐 쓸 수 있음
- `related`: 방향은 맞지만 수정량이 큼
- `miss`: 쓸 수 없음

배치 승격 조건:

- 목표 gap의 `candidate_coverage@5`가 증가한다.
- 기존 `useful` 컷이 `miss`로 내려가는 회귀가 0건이다.
- 목표가 아닌 컷에서 동일 family가 Top-5를 과도하게 채우지 않는다.
- 라이선스·provenance 기록률이 100%다.
- `product_bvh_export=yes` 포즈만 production pool에 들어간다. `delivery_mode=refined_only`이면
  기존 geometry 후보의 고객 전달 경로가 refine 결과만 반환하는지도 함께 확인한다. semantic 후보는
  refine 금지이므로 `delivery_mode=original`인 별도 selectable pool만 사용한다.

holdout은 gap을 고르거나 threshold를 조정하는 데 쓰지 않는다. 개발 컷으로 배치를 고른 뒤 마지막에
한 번만 확인한다.

---

## 7. 승격·버전·롤백

승격할 때 아래를 한 묶음으로 보존한다.

```text
pose-library-<version>/
├─ poses.db
├─ pose_semantics.db
├─ bvh/
├─ manifest.v1.jsonl
├─ manifest.csv
├─ semantic-build.json
├─ attribution.txt
├─ content-sha256.txt
└─ evaluation-summary.json
```

규칙:

- 기존 번들을 수정해서 덮어쓰지 말고 새 버전을 만든다.
- `poses.db`, `pose_semantics.db`, `bvh/`는 별도 파일이지만 서로 다른 `pose_library_version`을
  조합하지 않는다. semantic 기능을 끄는 rollout은 명시적 feature flag로만 수행한다.
- SQLite는 동기화 폴더에서 빌드하지 않는다.
- `provider`, `source_clip_id`, 필요한 `native_*_id`, `pose_family_id`, `license`,
  `product_bvh_export`, `delivery_mode`가 비어 있거나 검증 근거가 없으면 승격 실패다.
- 의미 검색을 켠 bundle은 `semantic-build.json`이 승인된 proposal/decision revision과
  `pose_library_version`, `pose_semantics.db` SHA-256, passage template와 전체 embedding 설정을
  고정하지 않으면 승격 실패다. 모델 weight가 앱 image/읽기 전용 volume에 있으면 해당 artifact
  hash와 license를 참조한다.
- attribution이 필요한 source가 하나라도 있으면 `attribution.txt`가 없을 때 승격 실패다.
- 장애가 생기면 `POSE_LIBRARY_URI`와 호환되는 semantic build/model artifact를 함께 이전 버전으로
  되돌린다. semantic만 실패하면 `/semantic-search`를 내리고 기존 geometry `/analyze`는 유지한다.

provisioner는 bundle을 live `data/`에 직접 풀지 않는다. 임시 version directory에 내려받아
`content-sha256.txt`, `pose_id → BVH hash`, geometry/semantic DB hash, encoder artifact 호환성을 모두
검사한 뒤 `current` pointer를 원자적으로 전환한다. `poses.db`가 이미 있다는 이유만으로 새 semantic
bundle download를 건너뛰면 안 된다. 이전 version directory는 한 묶음 롤백을 위해 보존한다.

공개 Git 저장소에는 source별 허용 여부와 관계없이 원본·변환 BVH, DB, 라이선스 계약 사본을 넣지
않는다. 서로 다른 조건의 파일이 한 폴더에 섞이는 실수를 막기 위한 저장소 공통 정책이다.

---

## 8. 현실적인 4주 실행안

### 1주차 — 기존 재고 정리

- `init_bvh_tag_inventory.py`로 1,308개 inventory와 654개 semantic unit을 재생성
- 현재 1,308 pose / 363 source clip의 원본 대응표 복구
- `provider`, `native_subject_id/asset_id/clip_id`, `original_title`, `source_clip_id`,
  `pose_family_id`, `license` 채움
- concrete pose별 posecode 측정값·typed atom 생성, 미러 좌우 swap 검증, P0/P1 태깅 검수
- 승인 text document를 렌더링하고 pinned multilingual E5 embedding/별도 semantic DB build
- `product_bvh_export`가 불명확한 포즈는 모두 `review`
- 고정 baseline DB·BVH·manifest hash 보존

**종료 조건:** 출처와 라이선스가 없는 기존 pose 0개, P0/P1 미검수 0개, 승인 revision과
`embedding_version`을 고정한 candidate `pose_semantics.db` 생성. 달성 전 신규 bulk import와 런타임
의미 검색 endpoint 구현 금지.

### 2주차 — CMU 스포츠 pilot

- 복싱·농구·무술·달리기에서 원본 clip 20개 선택
- clip당 대표 3개 → 눈검수 후 20~50 pose 유지
- 4-view audit, self-query, 고정 러프 전/후 비교

**종료 조건:** 목표 gap의 useful 후보가 늘고 기존 useful 회귀가 0건.

### 3주차 — Mixamo gap 보충

- CMU로 채워지지 않은 검술·태클·덩크 등 10~20 clip만 선택
- 같은 QA를 수행하고 `delivery_mode=refined_only`로 승격
- 원본·미조정 BVH 경로와 고객 전달용 refine 결과 경로를 분리

**종료 조건:** 품질 이득 확인 + 고객 전달 경로에서 refine 결과만 반환됨.

### 4주차 — ACCAD 호환성 pilot

- direct BVH 5~10개 파싱
- 관절명·축 mapping 차이 기록
- CC BY 3.0 attribution 산출물 생성
- 문제 없으면 다음 gap batch를 20 clip 이하로 확장

**종료 조건:** mapping 성공 100%, 4-view audit 통과, attribution 포함, 회귀 0건.

---

## 9. 현재 코드에서 먼저 메워야 할 운영 공백

아래는 가이드만으로 해결되지 않는 구현 항목이다.

1. **manifest 기반 import**: `manifest.v1.jsonl`을 읽어 `LibraryEntry.meta`와 DB의
   `source`·`license`를 실제 값으로 채워야 한다. CSV는 파생 뷰로만 둔다.
2. **license/export gate**: `product_bvh_export!=yes`인 pose는 고객 전달을 차단하고,
   `delivery_mode=refined_only`인 pose는 geometry refine 결과 경로로만 전달해야 한다. `/pose`와 export
   endpoint도 manifest를 서버에서 재검사하고, semantic selectable pool에는 `delivery_mode=original`만
   넣는다.
3. **전체 staging build**: 여러 source의 accepted BVH와 manifest를 합쳐 재현 가능하게 DB를
   빌드해야 한다.
4. **일반 BVH 대표 프레임 추출**: 현재 자동 대표 프레임 선택은 Mixamo FBX 전용이다.
5. **source별 관절 adapter**: ACCAD·AIST++·AI Hub를 쓸 경우 변환/매핑을 source별로 명시해야 한다.
6. **library version manifest**: DB·BVH·metadata·평가 결과 hash를 API health와 배포 로그에서
   확인할 수 있어야 한다.
7. ✅ **posecode/atom 생성기**: `src/posecode.py`와 `scripts/build_semantic_tagging.py`가 body-local
   축·연속 측정값·bucket·typed observed atom·mirror 검증을 버전된 결정적 코드로 생성한다.
8. ◐ **semantic resolver/validator**: 현재 structural validator와 review-only SQLite build는 구현됐다.
   다음으로 proposal+decision을 필드 단위로 합쳐 명시적으로 검수된 accepted
   값만 사용한다. 미검수 값은 drop/unknown 처리하고 orphan·set 역할 누락·vocab·버전 불일치에서
   fail-closed한다.
9. ✅ **dense index builder**: versioned passage, pinned multilingual E5 embedding, FTS5, typed atom을
   별도 `pose_semantics.db`와 `semantic-build.json`으로 재현 가능하게 만들고 전수 validator를
   통과시켰다. schema v2는 concrete member별 PoseCode 측정값 27개도 고정한다.
10. ◐ **로컬 query encoder**: 정확한 model/revision/hash load와 deterministic query embedding은
    구현했다. public server 투입 전 bounded concurrency와
    `(embedding_version, raw_query, constraints)` cache를 제공한다. silent model fallback은 금지한다.
11. ◐ **semantic runtime/API**: 내부 CLI와 `POST /semantic-search`의 dense+FTS, typed 3값 matcher,
    mirror resolver, build-aware cache, bounded concurrency, semantic health/version과
    `success | contextual_candidates | library_gap | clarification_required`를 구현했다.
    다음은 실제 다인 set resolver와 앱 검색 UI 연결이다.
12. **atomic bundle provisioner**: temp version directory에서 모든 hash/version을 검증하고 `current`
    pointer를 원자 전환한다. geometry·semantic DB·BVH·encoder를 서로 다른 release에서 섞지 않는다.
13. **refine authorization**: `/analyze` geometry 후보에만 서버 서명 selection token을 발급하고
    `/refine`에서 검증한다. semantic 후보에는 token을 발급하지 않으며 optional boolean 생략으로
    우회할 수 없게 한다.
14. **semantic readiness**: `SEMANTIC_REQUIRED=1`은 encoder/DB mismatch에서 startup/readiness 실패,
    `0`은 geometry health 유지+semantic 전용 alert/SLO로 구분한다.

특히 1번과 2번이 끝나기 전에는 신규 데이터로 candidate DB를 만들어 품질 실험은 할 수 있지만,
그 DB를 제품에 배포하지 않는다.

---

## 10. 매 배치 체크리스트

### 수집 전

- [ ] 해결할 실제 gap과 개발 러프가 정해졌다.
- [ ] 공식 source URL과 약관 원문을 저장했다.
- [ ] `commercial_use`, `raw_redistribution`, `product_bvh_export`, `delivery_mode`를 기록했다.
- [ ] 10~20 source clip 이하로 범위를 제한했다.

### 변환 후

- [ ] 원본 hash와 변환 recipe·프레임 번호가 manifest에 있다.
- [ ] 모든 keeper가 1-frame BVH다.
- [ ] body 관절 매핑·NaN·축 검사를 통과했다.
- [ ] 4-view sheet를 사람이 확인했다.
- [ ] 중복·무의미 프레임을 rejected 사유와 함께 분리했다.

### 승격 전

- [ ] active pose마다 inventory 1행과 semantic unit 1개가 있고 orphan·중복 hash가 해소됐다.
- [ ] 의미 검색 대상은 승인된 caption·alias·typed atom·posecode revision만 포함한다.
- [ ] 승인 unit마다 text document/embedding이 있고 dimension·finite·L2 norm·document hash·
  `embedding_version` 검사를 통과했다.
- [ ] geometry self-query가 `#1`, `dist=0.000`이고 semantic golden/holdout 결과가 저장됐다.
- [ ] 좌우 mirror 선택, 부정·정도·unknown, set 완전성, 문화 hallucination 회귀 검사를 통과했다.
- [ ] 목표 gap의 `candidate_coverage@5`가 증가했다.
- [ ] 기존 useful 컷의 회귀가 0건이다.
- [ ] provenance와 라이선스 기록률이 100%다.
- [ ] production pool에는 `product_bvh_export=yes`만 있고 `delivery_mode`가 실제 API 경로와 맞는다.
- [ ] attribution 파일과 bundle hash가 있다.
- [ ] 이전 library·semantic DB·encoder artifact 조합으로 롤백할 수 있다.

한 항목이라도 빠지면 `accepted`가 아니라 `candidate` 또는 `review` 상태로 남긴다.
