# Semantic embedding index build 보고서

> 생성일: 2026-08-18  
> 상태: pinned E5 + member PoseCode staging DB 생성 완료 · 내부 runtime/API development 통과 · holdout 승격 전

## 결과

| 항목 | 결과 |
|---|---:|
| semantic unit | 616 |
| pose member | 1232 |
| text document / embedding | 2892 / 2892 |
| observed unit atom | 5044 |
| member당 PoseCode 측정값 | 27 |
| embedding dimension | 384 |
| 최대 token 수 / truncation | 70 / 0 |
| validator | pass |
| 기존 build 재사용 | 아니오 |

## 고정한 encoder 계약

- model: `intfloat/multilingual-e5-small`
- revision: `fd1525a9fd15316a2d503bf26ab031a61d056e98`
- profile: `multilingual-e5-small-onnx-fp32-v1`
- embedding version: `multilingual-e5-small-onnx-fp32-v1:fd1525a9fd15316a2d503bf26ab031a61d056e98:passage-v1`
- dtype/dimension: `float32[384]`
- pooling: `attention_mask_mean`
- prefix: query=`query: `, passage=`passage: `
- normalization: L2 `True`
- runtime: onnxruntime `1.28.0`, tokenizers `0.23.1`

## 재현성

- semantic build ID: `sha256:217d56e31c42634ec920db8704dd151b088b2f12291d5e3726f5b34c9be50196`
- pose library version: `sha256:22eb5e9c24a954c11b68f684f327a71e42b694a9aed7e721589d30d84f724c76`
- search documents: `sha256:1972841eca19bc458429d445ba4dabe0aa5f7e1f740d63b56b28ff04792f0e5c`
- geometry inventory: `sha256:380e948317ce8259997b74eae5da0d5bcecb97e2fa3b714dec5f0c4d6f37fae7`
- geometry DB: `sha256:618fd1d51470ac376276a6af65bd153dac746d4c50740328df4576c80c8e2175`
- PoseCode proposals: `sha256:da8e647a7940f212a219ba1ae1117bf48ba1137192053e7763745dc3e3eb2f77`
- encoder artifacts: `sha256:510208f1e70828c800bfe64b397b85265ba14413d77290e7619f22c4cf987132`
- embedding matrix: `sha256:cdd04145fd889119197310baa1a54ba00f1b7a2a803f9fc1e626b358d8cccb7f`
- semantic DB: `sha256:72aaa4d57352a84f78fb8833742914f7164814ab922de1e59f42c5eae873cf84`

## 산출물

- build directory: `data/semantic/builds/217d56e31c42634ec920db8704dd151b088b2f12291d5e3726f5b34c9be50196`
- database: `data/semantic/builds/217d56e31c42634ec920db8704dd151b088b2f12291d5e3726f5b34c9be50196/pose_semantics.db`
- manifest: `data/semantic/builds/217d56e31c42634ec920db8704dd151b088b2f12291d5e3726f5b34c9be50196/semantic-build.json`
- model artifacts: `data/models/` 아래 로컬 캐시(Git 제외)
- official model card: https://huggingface.co/intfloat/multilingual-e5-small

## 승격 상태

이 DB는 재현 가능한 staging artifact다. golden v2 development와 semantic API/health는
통과했지만 holdout과 release bundle 승격 전이므로 `production_ready=false`다. 기존 geometry 검색에는 영향을 주지 않는다.
