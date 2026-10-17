"""문장 임베딩(dense) 인코더 — multilingual-e5-small ONNX를 새 의존성 없이 돌린다.

`src/semantic_search.py`의 `TextEncoder` 자리에 꽂는다. 기존 태그·글자 점수는 그대로 두고
'뜻 유사도' 성분(dense)을 하나 더하는 용도다(docs/SEMANTIC_SEARCH.md §3).

- 모델: Xenova/multilingual-e5-small `onnx/model_quantized.onnx`(int8, ~118MB, MIT) +
  `tokenizer.json`. 위치는 `SEMANTIC_EMBEDDING_DIR`(기본 data/models/multilingual-e5-small).
- 실행: 서버에 이미 있는 onnxruntime. 토크나이저는 `tokenizers` 패키지 없이 tokenizer.json의
  Unigram(SentencePiece) 모델을 직접 읽는다. 정규화는 NFKC로 근사한다(XLM-R의 precompiled
  charsmap과 드물게 다를 수 있다 — 검색 순위에는 영향이 작다).
- E5 규약: 쿼리는 "query: ", 문서는 "passage: " 접두어. mean pooling 후 L2 정규화.
- 라이브러리 임베딩은 한 번 계산해 캐시(npz)한다. 키 = 모델 해시 + 문서 해시.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

MODEL_FILE = "model_quantized.onnx"
TOKENIZER_FILE = "tokenizer.json"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
MAX_TOKENS = 128


class EmbeddingUnavailable(RuntimeError):
    """모델 파일이 없거나 열 수 없다."""


class UnigramTokenizer:
    """tokenizer.json(Unigram + Metaspace) → token ids. XLM-R/E5 계열용 최소 구현."""

    def __init__(self, spec: dict):
        model = spec.get("model") or {}
        if model.get("type") != "Unigram":
            raise EmbeddingUnavailable(f"unsupported tokenizer model: {model.get('type')}")
        vocab = model["vocab"]
        self.pieces = {piece: (i, float(score)) for i, (piece, score) in enumerate(vocab)}
        self.unk_id = int(model.get("unk_id") or 0)
        self.max_piece = max(len(p) for p in self.pieces)
        self.unk_score = min(score for _, score in vocab) - 10.0
        special = {t["content"]: t["id"] for t in spec.get("added_tokens") or []}
        self.bos_id = special.get("<s>", 0)
        self.eos_id = special.get("</s>", 2)
        self.pad_id = special.get("<pad>", 1)

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        return " ".join(text.split())

    def _encode_word(self, word: str) -> list[int]:
        """Viterbi: 점수 합이 가장 큰 조각 분할."""
        n = len(word)
        best = [-math.inf] * (n + 1)
        back: list[tuple[int, int]] = [(0, -1)] * (n + 1)
        best[0] = 0.0
        for end in range(1, n + 1):
            for start in range(max(0, end - self.max_piece), end):
                if best[start] == -math.inf:
                    continue
                piece = word[start:end]
                hit = self.pieces.get(piece)
                if hit is not None:
                    score = best[start] + hit[1]
                    if score > best[end]:
                        best[end], back[end] = score, (start, hit[0])
            if best[end] == -math.inf:  # 어떤 조각도 안 맞는 글자 → unk 한 글자
                best[end], back[end] = best[end - 1] + self.unk_score, (end - 1, self.unk_id)
        ids, pos = [], n
        while pos > 0:
            start, tid = back[pos]
            ids.append(tid)
            pos = start
        ids.reverse()
        merged = []  # 연속된 unk는 하나로(sentencepiece 동작)
        for tid in ids:
            if not (tid == self.unk_id and merged and merged[-1] == self.unk_id):
                merged.append(tid)
        return merged

    def encode(self, text: str, max_tokens: int = MAX_TOKENS) -> list[int]:
        words = self._normalize(text).split(" ")
        ids = [self.bos_id]
        for w in words:
            if w:
                ids.extend(self._encode_word("▁" + w))
        ids = ids[: max_tokens - 1]
        ids.append(self.eos_id)
        return ids


class E5OnnxEncoder:
    """`TextEncoder` 구현. fit(docs) → (n, d) 정규화 행렬, encode(q) → (1, d)."""

    name = "multilingual-e5-small-int8"

    def __init__(self, model_dir: str, *, cache_dir: Optional[str] = None, threads: int = 0):
        root = Path(model_dir)
        model_path, tok_path = root / MODEL_FILE, root / TOKENIZER_FILE
        if not model_path.is_file() or not tok_path.is_file():
            raise EmbeddingUnavailable(f"{MODEL_FILE}/{TOKENIZER_FILE} not found in {root}")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - 서버 기본 의존성
            raise EmbeddingUnavailable("onnxruntime not installed") from exc
        self.tokenizer = UnigramTokenizer(json.loads(tok_path.read_text(encoding="utf-8")))
        opts = ort.SessionOptions()
        ort.disable_telemetry_events()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.model_sha256 = _sha256(model_path)
        self.tokenizer_sha256 = _sha256(tok_path)
        self.cache_dir = Path(cache_dir) if cache_dir else root / "cache"

    # ── 배치 인코딩 ──
    def _embed(self, texts: Sequence[str], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            chunk = [self.tokenizer.encode(t) for t in texts[i:i + batch]]
            width = max(len(ids) for ids in chunk)
            ids = np.full((len(chunk), width), self.tokenizer.pad_id, dtype=np.int64)
            mask = np.zeros((len(chunk), width), dtype=np.int64)
            for r, row in enumerate(chunk):
                ids[r, :len(row)] = row
                mask[r, :len(row)] = 1
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]          # (b, t, d)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-6, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out.append(pooled.astype(np.float32))
        return np.vstack(out)

    def fit(self, documents: Sequence[str]) -> np.ndarray:
        texts = [PASSAGE_PREFIX + d for d in documents]
        key = hashlib.sha256(("\n".join(texts) + self.model_sha256 +
                              self.tokenizer_sha256 + str(MAX_TOKENS) +
                              "unigram-v1").encode("utf-8")).hexdigest()[:16]
        cache = self.cache_dir / f"passages-{key}.npy"
        if cache.is_file():
            matrix = np.load(cache)
            if matrix.ndim == 2 and matrix.shape[0] == len(texts) and np.isfinite(matrix).all():
                return matrix
        matrix = self._embed(texts)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache, matrix)
        except OSError:
            pass  # 캐시는 선택. 읽기 전용 볼륨이면 매번 계산한다.
        return matrix

    def encode(self, text: str) -> np.ndarray:
        return self._embed([QUERY_PREFIX + text])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dense_encoder(model_dir: str, **kwargs) -> E5OnnxEncoder:
    return E5OnnxEncoder(model_dir, **kwargs)


def model_dir_default(data_dir: str) -> str:
    return os.path.join(data_dir, "models", "multilingual-e5-small")
