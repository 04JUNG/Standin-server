# 포즈 라이브러리 번들과 버전 식별

운영 추론 서버는 기동할 때 `s3://<assets>/pose-library/v1.tar.gz`를 받아 푼다. key가 고정이고
`POSE_LIBRARY_VERSION`도 인프라에 `"v1"`로 고정돼 있어서, 지금은 BFF의
`analysis_candidates.pose_library_version`이 항상 `"v1"`이다. 어느 번들이 검색에 답했는지
알 수 없으니 라이브러리를 바꾼 전후를 비교할 수 없다.

그래서 번들 루트에 `library_manifest.json`을 둔다. 버전은 **내용 해시에서** 만든다
(`lib-YYYYMMDD-<content_sha256 앞 8자>`). 같은 내용이면 같은 버전이 나온다.

## 명령

```bash
# 1) 지금 S3에 있는 번들을 내용 그대로 다시 묶고 manifest만 붙인다. 검색 결과는 변하지 않는다.
python scripts/build_pose_bundle.py baseline \
  --source data/_s3/pose-library-v1.tar.gz --out data/bundles/baseline

# 2) 로컬 정리 DB(`python -m pose_curation publish` 결과)를 배포 형식으로 바꾼다.
#    정리 검수 서버가 DB를 잡고 있으면 열리지 않으니 먼저 서버를 내린다.
python scripts/build_pose_bundle.py curated \
  --curated-db data/curation/library/poses.db --curation-dir data/curation \
  --base data/_s3/pose-library-v1.tar.gz --out data/bundles/next --exclude-unresolved-rigs

# 3) 재생 게이트 보고서(JSON, status=passed|failed)를 manifest에 기록한다.
python scripts/build_pose_bundle.py record-gate --bundle data/bundles/next --report <report.json>

# 4) 검증만 → 배포
python scripts/deploy_pose_library.py data/bundles/next --dry-run
python scripts/deploy_pose_library.py data/bundles/next
```

`curated`가 하는 일:

| 정리 DB | 번들 |
|---|---|
| `bvh_path` = 로컬 절대경로 | `data/bvh/<파일>.bvh`. 기존 포즈는 원래 파일 이름, 신규 포즈는 `<pose_id>.bvh` |
| meta에 로컬 경로·검수 참고 러프·보정 입력 해시 | 허용 목록(`scripts/pose_bundle_policy.py::META_ALLOWLIST`)의 키만. 지운 값은 `VACUUM`으로 파일에서도 없앤다 |
| 신규 포즈 미리보기 512px 캐릭터 렌더 | 256×256 JPEG(q78). 렌더 뒤 해시가 바뀐 미리보기는 거부 |
| 기존 포즈 썸네일·BVH | 기준 번들에서 바이트 그대로 복사 |
| 출처 | `ATTRIBUTION.md`에 신규 포즈 출처(저작자·라이선스·원본)를 덧붙인다 |

빌더는 끝에서 배포 검증기를 그대로 돌리고, 통과한 번들에만 manifest를 쓴다.

## 배포 검증기가 막는 것

`deploy_pose_library.py`는 업로드 전에 아래를 모두 통과해야 진행한다.

- 서버 기동 조건: feature_version, view 값, 투영, BVH 파일, feature_blob, 썸네일
  (`thumbnails.thumbnail_filename` — `.jpg`. 예전 검증기만 `.png`를 찾던 불일치를 없앴다)
- **개인정보**: pose_id·meta·bvh_path에 설치·작업 ID, 사용자 입력 해시 접두사(`user_<sha12>`),
  BetaData key·버킷, 로컬 경로가 있으면 실패. 값 자체는 출력하지 않는다.
  자산 버킷은 버전 관리라 한 번 올라간 값은 이전 버전에 남는다.
- **리그 호환**: 운영 FBX 변환기(`converter.bone_map.resolve_profile`)가 모르는 리그는 실패.
  검색에는 나오는데 내보내기가 안 되는 포즈를 막는다. 100STYLE 리그는 #60
  (`converter/bone_map.py`의 `100style` 프로파일)이 병합된 뒤부터 통과한다. 그 전에 번들을
  만들면 `--exclude-unresolved-rigs`로 빼야 한다. 100STYLE 포즈는 refine 조정 없이 베이스로 나간다.
- **manifest**: 형식, `db_sha256`, `content_sha256`(번들 전체를 다시 해시)이 맞아야 한다.
  없으면 실패하고, 옛 번들을 꼭 올려야 할 때만 `--allow-no-manifest`.
- **재생 게이트**: `replay_gate.status`가 `passed`(또는 내용이 부모와 같은 `not_required`)여야
  한다. 건너뛰려면 `--skip-gate --reason "…"`이 필요하고, 근거가 배포 기록에 남는다.

`index.pkl`은 번들에 넣지 않는다. 서버는 읽지 않고(`scripts/run_demo.py`만 쓴다), 남겨 두면
DB와 어긋난 pickle이 섞인다. 반대로 `ATTRIBUTION.md`는 이제 넣는다(CC BY 표기 유지).

배포와 롤백은 `--log`(기본 `data/deploys.jsonl`)에 한 줄씩 남는다:
`library_version`, `content_sha256`, 새·직전 S3 `VersionId`, 안정화 결과, 게이트 건너뜀 근거.

## manifest 필드

| 필드 | 뜻 |
|---|---|
| `library_version` | `lib-YYYYMMDD-<hash8>`. 해시 부분은 `content_sha256` 앞 8자 |
| `content_sha256` | `poses.db`, `bvh/**`, `thumbs/**`, `ATTRIBUTION.md`의 `<경로>\t<sha256>` 줄을 경로순으로 이은 것의 SHA-256 |
| `db_sha256` | `poses.db` 해시 |
| `parent` | 이 번들의 바탕이 된 번들. manifest 없는 옛 번들이면 `v1`과 그 내용 해시 |
| `curation` | 만든 방식(baseline/curated), 정리 DB·manifest 해시, 그룹·리그별 수 |
| `replay_gate` | `not_run` → `record-gate`로 `passed`/`failed`. 보고서 해시·기준 버전 포함 |
| `privacy_scan`, `compat` | 빌더가 통과시킨 검사. `compat.excluded`에 리그 때문에 뺀 포즈 |

## 서버가 버전을 싣는 방식

추론 서버는 기동할 때 `poses.db` 옆의 manifest를 읽어 `CFG.pose_library_version`에 넣는다
(`api/app.py::_resolve_library_identity`). `/analyze`의 `inference_metadata`, `/refine`의 식별자,
`/healthz`의 `pose_library`가 모두 같은 값을 쓴다. manifest가 없으면 env 값으로 폴백하므로
옛 번들로 롤백해도 기동은 되고, 프로덕션에서 manifest가 DB와 맞지 않으면 기동을 막는다.

재생 게이트는 `python -m pose_gaps gate`(`docs/POSE_GAP_LOOP.md`)가 만든 보고서를
`record-gate`로 manifest에 기록한다.

## 로컬 실행 메모

- Windows에서 작은 파일 6천여 개를 다루므로 baseline 빌드는 1분 남짓 걸린다.
- 테스트의 임시 폴더 경로가 길면 Windows 260자 제한에 걸린다. 로컬에서는
  `pytest --basetemp=<짧은 경로>`를 쓴다(CI는 리눅스라 해당 없음).
