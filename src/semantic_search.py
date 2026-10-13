"""자연어 → 기존 라이브러리 포즈 검색(04J-285).

기하 검색(`src/search.py`)과 완전히 분리된 두 번째 검색 경로다. 러프에서 관절을 충분히
뽑지 못한 컷(04J-286 동적 라우팅)이나 작가가 직접 문장으로 찾을 때 쓴다.

입력은 한국어 문장 하나와 출처(`user` | `vlm`)다. 출력은 Top-K 후보와 각 후보의
포즈 JSON이며, 상태는 셋 중 하나다.

  ok         임계값 이상 후보가 있다.
  no_result  검색은 정상인데 임계값을 넘는 후보가 없다.
  error      색인·인코더·포즈 JSON·ID 정합성 같은 시스템 문제. ``error.code``에 원인.

호출자는 ``no_result``와 ``error``를 다르게 다뤄야 한다. ``no_result``는 "라이브러리에
그런 포즈를 현재 색인·조건·임계값에서 확인하지 못했다"는 정상 판정이고,
자산 부재의 증명은 아니다. ``error``는 "판정을 못 했다"는 뜻이다.

색인(태그 → 점수)
-----------------
04J-284 태깅 v2(`search-v2.json`)의 다섯 그룹을 쓴다.

1. **개념 일치**(concept): 쿼리에서 태그 어휘(라벨 + `config/semantic_lexicon.v1.json`의
   활용형·동의어)를 찾아 개념 목록을 만들고, 포즈가 그 개념 태그를 가졌는지 본다.
   필드별 가중치로 덮은 비율이 점수다. 근거가 태그 단위로 남는다.
2. **상·하체 묘사**(body): v2에는 상체·하체 전용 태그가 없다. 신체 묘사는 한 줄 설명에만
   있으므로, 쿼리를 신체 부위 단어로 자른 구절과 설명 문장의 글자 3-gram 포함률을
   상체·하체 따로 계산하고 평균한다. 신체 묘사가 없는 쿼리에서는 빠진다.
3. **문장 유사도**(text): 설명 + 태그 라벨의 글자 2·3-gram TF-IDF 코사인. 어휘에 없는
   표현을 받쳐 주는 안전망이다.

최종 점수는 쿼리에 실제로 있는 성분만 가중 평균한다(없는 성분은 분모에서 뺀다).
dense 임베딩은 이 배포 환경에 모델·네트워크가 없어 기본값에 넣지 않았다
(`TextEncoder`가 교체 지점이다 — docs/SEMANTIC_SEARCH.md §3).

불변식
------
* 후보 ``pose_id``와 포즈 JSON의 ``pose_id``는 같아야 한다. 다르거나 JSON이 없으면 그
  후보를 조용히 빼지 않고 요청 전체를 ``error``로 돌린다.
* 기하 검색의 후보·순위·refine 입력에는 아무 영향이 없다. 시맨틱 후보는 2D 관절 근거가
  없으므로 ``refine_allowed=False``를 고정한다.
* 격리(quarantine)된 포즈는 기하 검색과 똑같이 빠진다(`allowed_pose_ids`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Optional, Protocol, Sequence
from urllib.parse import quote

from .semantic_relevance import relevance_constraints

import numpy as np

SEMANTIC_CODE_VERSION = "semantic-search-v1"
POSE_RECORD_SCHEMA = "standin-pose-record-v1"
SOURCES = ("user", "vlm")
STATUS_OK = "ok"
STATUS_NO_RESULT = "no_result"
STATUS_ERROR = "error"
GROUP_KEYS = ("pose_action", "scene", "objects", "meaning")
THUMB_VIEWS = ("front", "three_quarter", "side", "back")
MIRROR_SUFFIX = "_mirror"
DISCRETE_ACTIONS = frozenset({"throw", "pass", "kick", "slash", "thrust", "strike",
                              "shoot_ball", "fire", "swim", "stretch", "cast_spell"})

#: 상·하체 구절을 자르는 신체 부위 단어. 길이 내림차순으로 매칭한다.
UPPER_BODY_TERMS = ("팔꿈치", "손목", "손바닥", "어깨", "상체", "가슴", "고개", "머리",
                    "허리", "양팔", "한팔", "양손", "한손", "주먹", "팔", "손", "몸통", "등", "몸")
LOWER_BODY_TERMS = ("발끝", "발목", "허벅지", "엉덩이", "골반", "하체", "양다리",
                    "한다리", "무릎", "다리", "발")


class ErrorCode:
    """``error.code`` 값. BFF·라우터가 분기할 수 있게 고정 문자열로 둔다."""

    INDEX_NOT_LOADED = "index_not_loaded"          # 기동 시 색인을 못 올렸다(설정 누락 등)
    INDEX_LOAD_FAILED = "index_load_failed"        # 파일 없음·JSON 깨짐·스키마 불일치
    INDEX_VERSION_MISMATCH = "index_version_mismatch"  # 태그가 다른 poses.db 기준
    INDEX_INCONSISTENT = "index_inconsistent"      # 태그에 있는데 DB에 없는 pose_id 등
    ENCODER_FAILED = "encoder_failed"              # 쿼리 인코딩(임베딩) 실패
    POSE_JSON_MISSING = "pose_json_missing"        # 후보의 포즈 JSON을 만들 수 없음
    POSE_ID_MISMATCH = "pose_id_mismatch"          # 후보 ID ≠ JSON 안의 pose_id
    INVALID_QUERY = "invalid_query"                # 빈 문장·너무 긴 문장
    INVALID_SOURCE = "invalid_source"              # user|vlm 외
    INVALID_TAGS = "invalid_tags"                  # tags에 태그 사전에 없는 값
    INTERNAL_ERROR = "internal_error"


class SemanticSearchError(RuntimeError):
    def __init__(self, code: str, message: str, detail: Optional[dict] = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.detail = detail or {}


# ─────────────────────────────── 설정 ───────────────────────────────


@dataclass(frozen=True)
class SemanticSettings:
    """검색 파라미터. 값은 `src/config.py`의 SEMANTIC_* env에서 온다."""

    top_k: int = 5
    max_top_k: int = 20
    min_score_user: float = 0.40
    min_score_vlm: float = 0.35
    weight_concept: float = 0.55
    weight_body: float = 0.20
    weight_text: float = 0.25
    weight_dense: float = 0.0        # 임베딩을 붙였을 때만 의미가 있다
    dense_lo: float = 0.78
    dense_hi: float = 0.90
    text_cos_ref: float = 0.45       # 이 코사인 이상이면 text 성분 1.0
    # 쿼리 단어 중 이 포즈가 설명하지 못하는 비율만큼 깎는다. 최종 = 기본 × (floor + (1-floor)×덮은 비율).
    # "냉장고 문을 여는"이 '문' 하나만 맞은 포즈에 만점을 주지 않게 한다.
    coverage_floor: float = 0.45
    # 라이브러리에 없는 단어 비율 u에 대해 점수 × (1-u)^power. 바닥값 없이 깎는다.
    unknown_word_power: float = 1.0
    body_manner_miss_weight: float = 0.5
    # 개념·신체 묘사를 하나도 못 읽고 글자 유사도만 남은 쿼리는 근거가 약하다.
    text_only_factor: float = 0.75
    collapse_mirrors: bool = True
    max_query_chars: int = 500
    near_miss_count: int = 3

    def min_score(self, source: str) -> float:
        return self.min_score_vlm if source == "vlm" else self.min_score_user

    def validate(self) -> None:
        if not 1 <= self.top_k <= self.max_top_k:
            raise ValueError("SEMANTIC_TOP_K must be in [1, SEMANTIC_MAX_TOP_K]")
        for name in ("min_score_user", "min_score_vlm"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"semantic {name} must be in [0,1]")
        weights = (self.weight_concept, self.weight_body, self.weight_text)
        if any(w < 0 or not math.isfinite(w) for w in weights) or sum(weights) <= 0:
            raise ValueError("semantic weights must be non-negative and not all zero")
        if self.text_cos_ref <= 0:
            raise ValueError("SEMANTIC_TEXT_COS_REF must be positive")
        if not 0.0 <= self.text_only_factor <= 1.0:
            raise ValueError("SEMANTIC_TEXT_ONLY_FACTOR must be in [0,1]")
        if self.unknown_word_power < 0:
            raise ValueError("SEMANTIC_UNKNOWN_WORD_POWER must be non-negative")
        if not 0.0 <= self.coverage_floor <= 1.0:
            raise ValueError("SEMANTIC_COVERAGE_FLOOR must be in [0,1]")


# ─────────────────────────────── 텍스트 처리 ───────────────────────────────


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣" or "ㄱ" <= ch <= "ㆎ"


def normalize_text(text: str) -> str:
    """NFKC + 소문자 + 기호 제거 + 공백 하나로."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^0-9a-z가-힣ㄱ-ㆎ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _nospace(text: str) -> str:
    return text.replace(" ", "")


def char_ngrams(text: str, ns: Sequence[int] = (2, 3)) -> Counter:
    s = _nospace(text)
    grams: Counter = Counter()
    for n in ns:
        for i in range(len(s) - n + 1):
            grams[s[i:i + n]] += 1
    return grams


class TextEncoder(Protocol):
    """문장 → 벡터. 라이브러리 문서는 fit 시점에, 쿼리는 요청마다 인코딩한다.

    기본 구현은 `CharNgramTfidf`다. dense 임베딩(E5·Gemini 등)으로 바꿀 때는 같은
    인터페이스를 구현하고 `SemanticIndex.build(..., encoder=...)`에 넘긴다.
    """

    name: str

    def fit(self, documents: Sequence[str]) -> np.ndarray: ...

    def encode(self, text: str) -> np.ndarray: ...


class CharNgramTfidf:
    """글자 2·3-gram TF-IDF(L2 정규화). 한국어 활용형에 형태소 분석기 없이 견딘다."""

    name = "char-ngram-tfidf-v1"

    def __init__(self, ns: Sequence[int] = (2, 3)):
        self.ns = tuple(ns)
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)

    def fit(self, documents: Sequence[str]):
        from scipy import sparse

        grams = [char_ngrams(normalize_text(d), self.ns) for d in documents]
        df: Counter = Counter()
        for g in grams:
            df.update(g.keys())
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(documents)
        self.idf = np.array(
            [math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(df)], dtype=np.float32
        )
        rows, cols, vals = [], [], []
        for r, g in enumerate(grams):
            for t, c in g.items():
                rows.append(r)
                cols.append(self.vocab[t])
                vals.append((1.0 + math.log(c)) * self.idf[self.vocab[t]])
        mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab)),
                                dtype=np.float32)
        norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1))).ravel()
        norms[norms == 0] = 1.0
        return sparse.diags(1.0 / norms) @ mat

    def encode(self, text: str):
        from scipy import sparse

        g = char_ngrams(normalize_text(text), self.ns)
        cols, vals = [], []
        for t, c in g.items():
            idx = self.vocab.get(t)
            if idx is not None:
                cols.append(idx)
                vals.append((1.0 + math.log(c)) * self.idf[idx])
        vec = sparse.csr_matrix((vals, ([0] * len(cols), cols)),
                                shape=(1, len(self.vocab)), dtype=np.float32)
        norm = math.sqrt(float(vec.multiply(vec).sum()))
        return vec / norm if norm > 0 else vec


# ─────────────────────────────── 한국어 활용 ───────────────────────────────

_VERB_FIELDS = frozenset({"action", "posture", "intent", "emotion"})
_LINK_PARTICLES = ("", "을", "를", "이", "가", "은", "는", "으로", "로", "에", "도")
_HANGUL_BASE, _JONG_N, _JONG_L = 0xAC00, 4, 8
# 모음 축약: 이+어→여(젖히어→젖혀), 오+아→와, 우+어→워, 으+어→어(쓰어→써)
_CONTRACT = {20: 6, 8: 9, 13: 14, 18: 4}


def _syl(ch: str):
    code = ord(ch) - _HANGUL_BASE
    if not 0 <= code < 11172:
        return None
    return code // 588, (code % 588) // 28, code % 28


def _compose(lead: int, vowel: int, tail: int) -> str:
    return chr(_HANGUL_BASE + lead * 588 + vowel * 28 + tail)


def conjugations(stem: str) -> set[str]:
    """동사 어간 → 흔한 활용형. 형태소 분석기 없이 검색 표현만 넓힌다(오타 교정은 하지 않는다).

    받침 없는 어간: 젖히 → 젖히는·젖히고·젖히며·젖힌·젖힐·젖혀·젖혀서
    받침 있는 어간: 젓 → 젓는·젓고·젓으며·젓은·젓어 / 꿇 → 꿇는·꿇고·꿇은·꿇어
    """
    out: set[str] = set()
    if not stem or _syl(stem[-1]) is None:
        return out
    lead, vowel, tail = _syl(stem[-1])
    head = stem[:-1]
    if tail == 0:
        for ending in ("는", "고", "며", "면", "던", "게", "지"):
            out.add(stem + ending)
        out.add(head + _compose(lead, vowel, _JONG_N))   # 숙인·젖힌
        out.add(head + _compose(lead, vowel, _JONG_L))   # 숙일·젖힐
        if vowel in _CONTRACT:
            contracted = head + _compose(lead, _CONTRACT[vowel], 0)
            out.update({contracted, contracted + "서", contracted + "도"})
        elif vowel in (0, 4, 1, 5):                          # 가·서·개·세: 어미 '아/어'가 흡수됨
            out.update({stem + "서"})
        if stem.endswith("하"):
            out.update({head + "해", head + "해서", head + "한", head + "할"})
        if stem.endswith("르") and len(stem) >= 2:        # 휘두르 → 휘둘러
            prev = _syl(stem[-2])
            if prev and prev[2] == 0:
                out.add(stem[:-2] + _compose(prev[0], prev[1], _JONG_L) + "러")
    else:
        for ending in ("는", "고", "으며", "은", "어", "아", "을", "던", "지"):
            out.add(stem + ending)
    return out


# ─────────────────────────────── 개념 사전 ───────────────────────────────


@dataclass(frozen=True)
class Concept:
    """쿼리에서 찾은 개념 하나. alternatives 중 하나라도 포즈에 있으면 만족."""

    surface: str
    alternatives: frozenset  # {(field, value)}
    weight: float
    miss_weight: float = -1.0  # 포즈가 이 개념을 못 가졌을 때 분모에 더할 무게(<0이면 weight)
    word_index: int = -1       # 쿼리에서 처음 나온 단어 위치(상·하체 구절 안인지 판정용)

    @property
    def penalty(self) -> float:
        return self.weight if self.miss_weight < 0 else self.miss_weight

    def to_dict(self, labels: Mapping[tuple, str]) -> dict:
        return {
            "surface": self.surface,
            "weight": round(self.weight, 3),
            "tags": [
                {"field": f, "value": v, "label": labels.get((f, v), v)}
                for f, v in sorted(self.alternatives)
            ],
        }


class ConceptLexicon:
    """태그 라벨 + 동의어 → 쿼리 표현 사전. 긴 표현부터 겹치지 않게 매칭한다."""

    def __init__(self, taxonomy: dict, lexicon: dict, mined: Optional[dict] = None):
        vocab = taxonomy.get("vocabulary") or {}
        if not vocab:
            raise SemanticSearchError(ErrorCode.INDEX_LOAD_FAILED, "taxonomy has no vocabulary")
        self.field_weights: dict[str, float] = {
            k: float(v) for k, v in (lexicon.get("field_weights") or {}).items()
        }
        # 태그가 비어 있으면 '없음'이 아니라 '미확인'인 필드(도구·의도 등)는 못 맞춰도 덜 깎는다.
        self.field_miss_weights: dict[str, float] = {
            k: float(v) for k, v in (lexicon.get("field_miss_weights") or {}).items()
        }
        # 부분 위치를 가리지 않는 명사형 필드. 나머지(동작·자세·의도…)는 단어 첫머리에서만 맞춘다
        # ('내밀어'의 '밀어', '옮기는'의 '기는'을 막는다).
        self.anywhere_fields = set(lexicon.get("match_anywhere_fields") or ["object", "activity", "domain"])
        self.implies: dict[tuple, set] = {}
        for raw, targets in (lexicon.get("implies") or {}).items():
            f, _, v = raw.partition(":")
            self.implies.setdefault((f, v), set()).update(
                tuple(t.split(":", 1)) for t in targets)
        self.single_char_allowed = set(lexicon.get("single_char_allowed") or [])
        self.particles = set(lexicon.get("particles") or "")
        self.stopwords = tuple(sorted(lexicon.get("stopwords") or [], key=len, reverse=True))
        # 라벨을 '·'로 쪼개거나 어간 활용형을 만들 때 생기는 엉뚱한 표현('들기·잡기'의 '들고'는
        # 팔을 '들고'와 구분되지 않는다, '고개·몸 숙여 인사'의 '고개')을 막는다.
        blocked = {_nospace(normalize_text(w)) for w in lexicon.get("blocked_surfaces") or []}
        self.labels: dict[tuple, str] = {}
        self.field_group: dict[str, str] = {}
        for g in taxonomy.get("groups") or []:
            for f in g.get("fields") or []:
                self.field_group[f] = g["key"]
        surfaces: dict[str, set] = {}
        verb_stems: list[tuple[str, tuple]] = []   # 활용형을 만들 어간

        def add(surface: str, key: tuple):
            s = _nospace(normalize_text(surface))
            if not s or s in blocked:
                return
            if len(s) == 1 and s not in self.single_char_allowed:
                return
            surfaces.setdefault(s, set()).add(key)

        for f, values in vocab.items():
            if f not in self.field_weights:
                continue  # 가중치가 없는 필드는 검색 개념으로 쓰지 않는다
            for v, label in values.items():
                key = (f, v)
                self.labels[key] = label
                for part in [label, *label.split("·")]:
                    add(part, key)
                    p = _nospace(part)
                    if p.endswith("기") and len(p) >= 3:
                        add(p[:-1], key)  # 던지기 → 던지(던지는·던지고)
                        verb_stems.append((p[:-1], key))
                    elif p.endswith("기") and len(p) == 2:
                        # 한 글자 어간(뻗기·걷기·밀기)은 단독 매칭이 위험해 활용형만 넣는다.
                        verb_stems.append((p[0], key))
                    words_ = part.split(" ")
                    if len(words_) >= 2 and words_[-1].endswith("기") and len(words_[-1]) >= 2:
                        # '노 젓기'·'몸 낮추기' → '노를 젓는'·'몸을 낮춘'(사이 조사 허용)
                        head = "".join(words_[:-1])
                        stem = words_[-1][:-1]
                        for particle in _LINK_PARTICLES:
                            verb_stems.append((head + particle + stem, key))
        # 어휘에서 빠진 값의 동의어는 버리고 기록한다. 태그 사전이 바뀌어도 검색 전체를
        # 멈추지는 않되, /healthz identity에 남겨 사전 갱신을 놓치지 않게 한다.
        self.skipped_keys: list[str] = []
        for raw_key, words in (lexicon.get("synonyms") or {}).items():
            f, _, v = raw_key.partition(":")
            if (f, v) not in self.labels:
                self.skipped_keys.append(raw_key)
                continue
            for w in words:
                add(w, (f, v))
                if f in _VERB_FIELDS:
                    verb_stems.append((_nospace(normalize_text(w)), (f, v)))
        for stem, key in verb_stems:
            for form in conjugations(stem):
                add(form, key)
            if len(stem) >= 2:
                add(stem, key)
        # 말뭉치에서 캔 표현(scripts/mine_semantic_lexicon.py). 손으로 만든 표현을 덮지 않고,
        # 한 표현은 가장 특이한(lift 최대) 태그 하나에만 붙인다 — 여러 태그를 OR로 묶으면
        # '평영'이 눕기 포즈 아무것에나 맞는다.
        self.mined_count = 0
        best: dict[str, tuple] = {}
        for raw_key, entries in ((mined or {}).get("evidence") or {}).items():
            f, _, v = raw_key.partition(":")
            if (f, v) not in self.labels:
                continue
            for e in entries:
                term = _nospace(normalize_text(e.get("term", "")))
                if not term or term in surfaces or term in blocked:
                    continue
                rank = (e.get("lift", 0.0), e.get("precision", 0.0))
                if term not in best or rank > best[term][0]:
                    best[term] = (rank, (f, v))
        for term, (_, key) in best.items():
            forms = {term} | (conjugations(term) if key[0] in _VERB_FIELDS else set())
            for form in forms:
                if form not in surfaces:
                    add(form, key)
            self.mined_count += 1
        self.surfaces = {s: frozenset(k) for s, k in surfaces.items()}
        self._ordered = sorted(self.surfaces, key=lambda s: (-len(s), s))
        self._max_len = max(len(x) for x in self.surfaces)

    def weight_of(self, alternatives: Iterable[tuple]) -> float:
        return max(self.field_weights.get(f, 0.0) for f, _ in alternatives)

    def miss_weight_of(self, alternatives: Iterable[tuple]) -> float:
        return max(self.field_miss_weights.get(f, self.field_weights.get(f, 0.0))
                   for f, _ in alternatives)

    def expand_tags(self, tags: Iterable[tuple]) -> frozenset:
        """함의 확장: 인사(bow)는 숙이기(bend)를 포함한다 등. 매칭에만 쓴다(포즈 JSON은 원본)."""
        out = set(tags)
        frontier = list(out)
        while frontier:
            for t in self.implies.get(frontier.pop(), ()):
                if t not in out:
                    out.add(t)
                    frontier.append(t)
        return frozenset(out)

    def strip_stopwords(self, text: str) -> str:
        for w in self.stopwords:
            text = text.replace(w, " ")
        return re.sub(r"\s+", " ", text).strip()

    def _allowed_at(self, surface: str, s: str, i: int, word_start: list[bool]) -> bool:
        alts = self.surfaces[surface]
        if not word_start[i] and not any(f in self.anywhere_fields for f, _ in alts):
            return False
        if len(surface) == 1:
            if not word_start[i]:
                return False  # 장애물/건물의 '물'은 수영 대상이 아니다. 복합어는 명시 사전으로.
            end = i + 1
            if end < len(s) and _is_hangul(s[end]) and s[end] not in self.particles \
                    and not word_start[end]:
                return False
        return True

    def extract(self, query: str) -> list[Concept]:
        """왼쪽부터 가장 긴 표현을 고른다(leftmost-longest). 공백은 무시하되 단어 경계는 기억한다."""
        spaced = normalize_text(query)
        s, word_start, word_of = "", [], []
        w = 0
        for k, ch in enumerate(spaced):
            if ch == " ":
                w += 1
                continue
            word_start.append(k == 0 or spaced[k - 1] == " ")
            word_of.append(w)
            s += ch
        found: dict[frozenset, Concept] = {}
        i = 0
        while i < len(s):
            best = None
            for length in range(min(self._max_len, len(s) - i), 0, -1):
                cand = s[i:i + length]
                if cand in self.surfaces and self._allowed_at(cand, s, i, word_start):
                    best = cand
                    break
            if best is None:
                i += 1
                continue
            # 긴 표현은 같은 자리에서 시작하는 짧은 어간의 뜻도 갖는다('때리는' ⊃ '때리').
            alts = set(self.surfaces[best])
            for length in range(1, len(best)):
                stem = s[i:i + length]
                if stem in self.surfaces and self._allowed_at(stem, s, i, word_start):
                    alts |= self.surfaces[stem]
            alts = frozenset(alts)
            if alts not in found:
                found[alts] = Concept(best, alts, self.weight_of(alts), self.miss_weight_of(alts),
                                      word_of[i])
            i += len(best)
        return list(found.values())


def query_words(norm: str, lex: "ConceptLexicon") -> list[tuple[str, str]]:
    """덮음 판정에 쓸 쿼리 내용어와 그 앞 2글자 어간. 불용어·한 글자 조사는 뺀다."""
    out = []
    stop = set(lex.stopwords)
    for w in norm.split(" "):
        if not w or w in stop:
            continue
        core = w[:-1] if len(w) >= 3 and w[-1] in lex.particles else w
        if core in stop:
            continue
        if len(core) < 2 and core not in lex.single_char_allowed:
            continue
        if _body_region(core):  # 왼팔·양손 → 팔·손 (좌우는 설명에 거의 없다)
            for prefix in ("오른", "왼", "양", "한", "두"):
                if core.startswith(prefix) and len(core) > len(prefix):
                    core = core[len(prefix):]
                    break
        if len(core) == 2 and core[1] in _CONJ_ENDINGS:
            stem = core[0]          # 벌린·숙인·뻗고 → 벌·숙·뻗
        else:
            stem = core[:2]
        out.append((w, stem))
    return out


def _mask_stopwords(norm: str, lex: "ConceptLexicon") -> str:
    """불용어 단어를 자리표시자로 바꾼다(단어 위치는 유지해 상·하체 구절과 맞춘다)."""
    stop = set(lex.stopwords)
    out = []
    for w in norm.split(" "):
        core = w[:-1] if len(w) >= 3 and w[-1] in lex.particles else w
        out.append("x" if (w in stop or core in stop) else w)
    return " ".join(out)


def _stem_in_doc(stem: str, doc: "PoseDoc") -> bool:
    if len(stem) >= 2:
        return stem in doc.text_ns
    return any(w.startswith(stem) for w in doc.desc_words)  # '타' ↛ '타격', '벌' → '벌려'


_CONJ_ENDINGS = frozenset("는은고게며어아여린된진운인한친킨든던")


# ─────────────────────────────── 상·하체 구절 ───────────────────────────────


#: 신체 부위 앞에 붙어 그 부위를 꾸미는 단어. 앞 구절이 아니라 뒤 구절에 속한다.
BODY_MODIFIERS = frozenset({"한쪽", "양쪽", "왼쪽", "오른쪽", "왼", "오른", "두", "한", "양",
                            "앞", "뒤", "뒷", "앞쪽", "뒤쪽"})
_BODY_TERMS = sorted([(t, "upper") for t in UPPER_BODY_TERMS]
                     + [(t, "lower") for t in LOWER_BODY_TERMS], key=lambda x: -len(x[0]))


def _body_region(word: str) -> Optional[str]:
    """단어가 신체 부위로 시작하면 그 영역. '팔을'·'무릎으로'는 맞고 '발레'·'등반'은 아니다."""
    for prefix in ("오른", "왼", "양", "한", "두", ""):  # 왼팔·오른발·양손처럼 붙여 쓴 경우
        if prefix and not word.startswith(prefix):
            continue
        rest_word = word[len(prefix):]
        for term, region in _BODY_TERMS:
            if rest_word.startswith(term):
                rest = rest_word[len(term):]
                if not rest or rest[0] in "을를이가은는에의도만과와으로" or len(term) >= 2:
                    return region
    return None


#: 구절을 끝내는 활용 어미(연결·관형). "들고"·"들어"·"벌려"·"굽힌"에서 신체 묘사가 끝난다.
_CLAUSE_END = frozenset("고며서어아여려워해쳐혀겨는은린힌인운한친진든된선간던져")


def body_spans(query: str) -> list[tuple[int, int, str]]:
    """신체 묘사 구절의 (시작 단어, 끝 단어, 영역). 끝 단어는 포함한다.

    신체 부위 단어에서 시작해 그 뒤 첫 활용형 단어에서 끝난다. 그다음 말("검을 내려치려")은
    신체 모양이 아니라 행동이라 구절에 넣지 않는다.
    """
    words = [w for w in normalize_text(query).split(" ") if w]
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(words):
        region = _body_region(words[i])
        if not region:
            i += 1
            continue
        start = i - 1 if i > 0 and words[i - 1] in BODY_MODIFIERS and \
            (not spans or i - 1 > spans[-1][1]) else i
        end = i
        j = i + 1
        while j < len(words):
            if _body_region(words[j]) or (words[j] in BODY_MODIFIERS and j + 1 < len(words)
                                          and _body_region(words[j + 1])):
                if _body_region(words[j]) and _body_region(words[j]) == region:
                    end = j          # "오른팔을 머리 위로 들어"처럼 같은 영역이 이어지면 한 구절
                    j += 1
                    continue
                break                # 다른 영역(상체→하체)이면 새 구절
            end = j
            if len(words[j]) >= 2 and words[j][-1] in _CLAUSE_END:
                break
            j += 1
        spans.append((start, end, region))
        i = end + 1
    return spans


def body_clauses(query: str) -> dict[str, list[str]]:
    """쿼리를 신체 부위 단어 기준으로 잘라 상체·하체 구절로 나눈다(단어 경계 기준).

    "양팔을 위로 들고 한쪽 무릎을 굽힌" → upper=["양팔을위로들고"], lower=["한쪽무릎을굽힌"]
    """
    words = [w for w in normalize_text(query).split(" ") if w]
    out: dict[str, list[str]] = {"upper": [], "lower": []}
    for start, end, region in body_spans(query):
        clause = "".join(w for w in words[start:end + 1] if w not in ("자세", "포즈", "모습"))
        if len(clause) >= 2:
            out[region].append(clause)
    return out


def _containment(clause: str, doc_grams: set) -> float:
    grams = set(char_ngrams(clause, (2,)))
    if not grams:
        return 0.0
    return len(grams & doc_grams) / len(grams)


# ─────────────────────────────── 색인 ───────────────────────────────


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: str, what: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise SemanticSearchError(ErrorCode.INDEX_LOAD_FAILED, f"{what} not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticSearchError(ErrorCode.INDEX_LOAD_FAILED, f"{what} unreadable: {exc}") from exc


_META_KEEP = ("category", "category_label", "style", "pose_family_id",
              "curation_group", "review_status")


def _load_db_records(db_path: str) -> dict[str, dict]:
    if not os.path.exists(db_path):
        raise SemanticSearchError(ErrorCode.INDEX_LOAD_FAILED, f"poses.db not found: {db_path}")
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT pose_id, source, license, set_id, set_role, meta_json FROM poses"
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise SemanticSearchError(ErrorCode.INDEX_LOAD_FAILED, f"poses.db unreadable: {exc}") from exc
    out = {}
    for pose_id, source, license_, set_id, set_role, meta_json in rows:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except json.JSONDecodeError:
            meta = {}
        out[pose_id] = {
            "pose_id": pose_id,
            "source": meta.get("source") or source,
            "license": meta.get("license") or license_,
            "set_id": set_id,
            "set_role": set_role,
            "meta": {k: meta[k] for k in _META_KEEP if k in meta},
        }
    return out


@dataclass
class PoseDoc:
    pose_id: str
    tags: frozenset              # {(field, value)}
    groups: dict                 # 원본 그룹 구조(포즈 JSON용)
    description: str
    desc_grams: set = field(default_factory=set)
    text_ns: str = ""            # 설명 + 태그 라벨(공백 제거) — 쿼리 단어 덮음 판정용
    desc_words: tuple = ()       # 설명 단어 — 한 글자 어간은 단어 첫머리에서만 인정
    match_tags: frozenset = frozenset()  # tags + 함의 확장(매칭 전용)


@dataclass
class SemanticIndex:
    docs: list[PoseDoc]
    lexicon: ConceptLexicon
    encoder: TextEncoder
    doc_matrix: object
    pose_records: dict[str, dict]
    identity: dict
    thumbs_dir: Optional[str] = None
    family: dict[str, str] = field(default_factory=dict)
    dense_encoder: Optional[object] = None    # 선택: 문장 임베딩(src/semantic_embedding.py)
    dense_matrix: Optional[np.ndarray] = None
    review_preferences: dict = field(default_factory=dict)  # optional explicit local feedback

    @property
    def pose_count(self) -> int:
        return len(self.docs)

    @classmethod
    def load(
        cls,
        *,
        tags_path: str,
        taxonomy_path: str,
        lexicon_path: str,
        db_path: str,
        allowed_pose_ids: Optional[Iterable[str]] = None,
        excluded_pose_ids: Optional[Iterable[str]] = None,
        verify_db_sha256: bool = True,
        thumbs_dir: Optional[str] = None,
        encoder: Optional[TextEncoder] = None,
        dense_encoder: Optional[TextEncoder] = None,
        library_version: Optional[str] = None,
        review_preferences_path: Optional[str] = None,
    ) -> "SemanticIndex":
        tags = _read_json(tags_path, "semantic tags")
        taxonomy = _read_json(taxonomy_path, "semantic taxonomy")
        lexicon_raw = _read_json(lexicon_path, "semantic lexicon")
        mined_raw = None
        mined_name = lexicon_raw.get("mined_lexicon")
        if mined_name:
            mined_path = os.path.join(os.path.dirname(os.path.abspath(lexicon_path)), mined_name)
            mined_raw = _read_json(mined_path, "mined lexicon")
        if not isinstance(tags.get("poses"), list):
            raise SemanticSearchError(ErrorCode.INDEX_LOAD_FAILED, "tags file has no 'poses' list")
        if str(tags.get("version")) != str(taxonomy.get("version")):
            raise SemanticSearchError(
                ErrorCode.INDEX_LOAD_FAILED,
                f"tags version {tags.get('version')} != taxonomy {taxonomy.get('version')}",
            )
        db_sha = file_sha256(db_path) if os.path.exists(db_path) else None
        expected = tags.get("source_db_sha256")
        if verify_db_sha256 and expected and db_sha and expected != db_sha:
            raise SemanticSearchError(
                ErrorCode.INDEX_VERSION_MISMATCH,
                "semantic tags were built for a different poses.db",
                {"tags_source_db_sha256": expected, "db_sha256": db_sha},
            )
        records = _load_db_records(db_path)
        lexicon = ConceptLexicon(taxonomy, lexicon_raw, mined_raw)
        allowed = set(allowed_pose_ids) if allowed_pose_ids is not None else None
        excluded_ids = set(excluded_pose_ids or ())

        docs: list[PoseDoc] = []
        seen: set[str] = set()
        unknown: list[str] = []
        excluded = 0
        for item in tags["poses"]:
            pose_id = item.get("pose_id")
            if not isinstance(pose_id, str) or not pose_id:
                raise SemanticSearchError(ErrorCode.INDEX_INCONSISTENT, "tag entry without pose_id")
            if pose_id in seen:
                raise SemanticSearchError(ErrorCode.INDEX_INCONSISTENT,
                                          f"duplicate pose_id in tags: {pose_id}")
            seen.add(pose_id)
            if pose_id not in records:
                unknown.append(pose_id)
                continue
            if pose_id in excluded_ids or (allowed is not None and pose_id not in allowed):
                excluded += 1
                continue
            groups = item.get("groups") or {}
            tagset = set()
            clean_groups = {}
            for g in GROUP_KEYS:
                entries = []
                for t in groups.get(g) or []:
                    f, v = t.get("field"), t.get("value")
                    if f and v:
                        tagset.add((f, v))
                        entries.append({"field": f, "value": v,
                                        "label": t.get("label") or lexicon.labels.get((f, v), v)})
                clean_groups[g] = entries
            desc = ((item.get("description") or {}).get("text") or "").strip()
            label_text = " ".join(e["label"] for g in GROUP_KEYS for e in clean_groups[g])
            docs.append(PoseDoc(pose_id, frozenset(tagset), clean_groups, desc,
                                set(char_ngrams(normalize_text(desc), (2,))),
                                _nospace(normalize_text(f"{desc} {label_text}")),
                                tuple(normalize_text(desc).split(" ")),
                                lexicon.expand_tags(tagset)))
        if unknown:
            raise SemanticSearchError(
                ErrorCode.INDEX_INCONSISTENT,
                f"{len(unknown)} tagged pose_id(s) missing from poses.db",
                {"examples": unknown[:10]},
            )
        if not docs:
            raise SemanticSearchError(ErrorCode.INDEX_LOAD_FAILED, "semantic index is empty")

        encoder = encoder or CharNgramTfidf()
        texts = [
            f"{d.description} " + " ".join(lexicon.labels.get(t, t[1]) for t in sorted(d.tags))
            for d in docs
        ]
        try:
            matrix = encoder.fit(texts)
        except Exception as exc:  # noqa: BLE001 — 인코더 종류와 무관하게 원인 코드로 감싼다
            raise SemanticSearchError(ErrorCode.ENCODER_FAILED, f"index encoding failed: {exc}") from exc

        ids = {d.pose_id for d in docs}
        family = {}
        for d in docs:
            base = d.pose_id[: -len(MIRROR_SUFFIX)] if d.pose_id.endswith(MIRROR_SUFFIX) else d.pose_id
            family[d.pose_id] = base if base in ids else d.pose_id

        identity = {
            "code_version": SEMANTIC_CODE_VERSION,
            "tags_version": str(tags.get("version")),
            "tags_sha256": file_sha256(tags_path),
            "lexicon_version": lexicon_raw.get("version"),
            "lexicon_skipped_keys": list(lexicon.skipped_keys),
            "mined_lexicon": (mined_raw or {}).get("version"),
            "mined_terms": lexicon.mined_count,
            # 캔 사전이 지금 태그 파일로 만든 것이 아니면 표시한다(다시 캐야 한다).
            "mined_lexicon_stale": bool(mined_raw) and (
                mined_raw.get("source_tags_sha256") != file_sha256(tags_path)),
            "encoder": encoder.name,
            "db_sha256": db_sha,
            "library_version": library_version,
            "pose_count": len(docs),
            "excluded_pose_count": excluded,
        }
        dense_matrix = None
        if dense_encoder is not None:
            try:
                dense_matrix = np.asarray(dense_encoder.fit(texts), dtype=np.float32)
                if dense_matrix.ndim != 2 or dense_matrix.shape[0] != len(docs) or not np.isfinite(dense_matrix).all():
                    raise ValueError("dense document vectors have invalid shape or non-finite values")
            except Exception as exc:  # noqa: BLE001
                raise SemanticSearchError(ErrorCode.ENCODER_FAILED,
                                          f"dense index encoding failed: {exc}") from exc
            identity["dense_encoder"] = dense_encoder.name
            identity["dense_model_sha256"] = getattr(dense_encoder, "model_sha256", None)
        index = cls(docs, lexicon, encoder, matrix, records, identity, thumbs_dir, family,
                    dense_encoder, dense_matrix)
        if review_preferences_path:
            from .semantic_relevance import load_review_preferences
            load_review_preferences(index, review_preferences_path, normalize_text)
        return index

    # ── 포즈 JSON ──

    def pose_json(self, pose_id: str) -> Optional[dict]:
        record = self.pose_records.get(pose_id)
        doc = self._doc_by_id().get(pose_id)
        if record is None or doc is None:
            return None
        encoded = quote(pose_id, safe="")
        thumbs = {}
        for view in THUMB_VIEWS:
            if self.thumbs_dir and os.path.exists(
                os.path.join(self.thumbs_dir, f"{pose_id}__{view}.jpg")
            ):
                thumbs[view] = f"/pose/{encoded}/thumbnail?view={view}"
        return {
            "schema": POSE_RECORD_SCHEMA,
            "pose_id": record["pose_id"],
            "source": record["source"],
            "license": record["license"],
            "set_id": record["set_id"],
            "set_role": record["set_role"],
            "meta": dict(record["meta"]),
            "tags": {g: [dict(t) for t in doc.groups.get(g, [])] for g in GROUP_KEYS},
            "description": doc.description,
            "bvh_url": f"/pose/{encoded}/bvh",
            "thumbnail_urls": thumbs,
            "library": {
                "library_version": self.identity.get("library_version"),
                "db_sha256": self.identity.get("db_sha256"),
                "tags_version": self.identity.get("tags_version"),
            },
        }

    def _doc_by_id(self) -> dict[str, PoseDoc]:
        cache = self.__dict__.get("_doc_cache")
        if cache is None:
            cache = {d.pose_id: d for d in self.docs}
            self.__dict__["_doc_cache"] = cache
        return cache


# ─────────────────────────────── 검색 ───────────────────────────────


def _error(code: str, message: str, *, query: Optional[dict] = None,
           detail: Optional[dict] = None, started: Optional[float] = None,
           index: Optional[SemanticIndex] = None) -> dict:
    return {
        "status": STATUS_ERROR,
        "error": {"code": code, "message": message, "detail": detail or {}},
        "query": query,
        "candidates": [],
        "near_misses": [],
        "threshold": None,
        "top_k": None,
        "index": index.identity if index else None,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2) if started else 0.0,
    }


def semantic_search(
    index: Optional[SemanticIndex],
    text: str,
    *,
    source: str = "user",
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    settings: SemanticSettings = SemanticSettings(),
    tags: Optional[Sequence[str]] = None,
) -> dict:
    """자연어 한 문장 → 시맨틱 후보. **예외를 던지지 않는다** — 실패는 ``status=error``.

    Args:
        index: `SemanticIndex.load` 결과. None이면 ``index_not_loaded``.
        text: 한국어 문장(사용자 입력 또는 VLM이 러프에서 만든 설명).
        source: ``user`` | ``vlm``. 임계값 선택과 로그 구분에 쓴다.
        top_k: 반환 후보 수. None이면 ``settings.top_k``. ``settings.max_top_k``로 자른다.
        min_score: 임계값 덮어쓰기(평가용). None이면 source별 설정값.
        tags: 선택. 태그 사전 값 ``"field:value"`` 목록(예: VLM이 러프에서 고른 ``"action:throw"``).
            문장 해석보다 우선하는 확정 개념으로 쓴다. tags가 있으면 text는 비어도 된다.

    Returns:
        JSON 직렬화 가능한 dict. 키: status, error, query, candidates, near_misses,
        threshold, top_k, index, elapsed_ms.
    """
    started = time.perf_counter()
    if source not in SOURCES:
        return _error(ErrorCode.INVALID_SOURCE, f"source must be one of {SOURCES}",
                      started=started, index=index)
    norm = normalize_text(text if isinstance(text, str) else "")
    tag_list = list(tags or [])
    query_info = {"text": text, "source": source, "normalized": norm, "tags": tag_list}
    if not norm and not tag_list:
        return _error(ErrorCode.INVALID_QUERY, "query is empty", query=query_info,
                      started=started, index=index)
    if any(not isinstance(t, str) or ":" not in t for t in tag_list) or len(tag_list) > 30:
        return _error(ErrorCode.INVALID_TAGS, "tags must be up to 30 'field:value' strings",
                      query=query_info, started=started, index=index)
    if len(norm) > settings.max_query_chars:
        return _error(ErrorCode.INVALID_QUERY,
                      f"query longer than {settings.max_query_chars} chars",
                      query=query_info, started=started, index=index)
    if index is None:
        return _error(ErrorCode.INDEX_NOT_LOADED, "semantic index is not loaded",
                      query=query_info, started=started)
    if tag_list:
        unknown = [t for t in tag_list if tuple(t.split(":", 1)) not in index.lexicon.labels]
        if unknown:
            return _error(ErrorCode.INVALID_TAGS, "tags not in taxonomy",
                          query=query_info, detail={"unknown": unknown[:10]},
                          started=started, index=index)
    try:
        return _search(index, text, norm, source, top_k, min_score, settings, started,
                       query_info, tag_list)
    except SemanticSearchError as exc:
        return _error(exc.code, exc.message, query=query_info, detail=exc.detail,
                      started=started, index=index)
    except Exception as exc:  # noqa: BLE001 — 계약: 호출자에게 예외 대신 error 상태
        return _error(ErrorCode.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}",
                      query=query_info, started=started, index=index)


def _search(index, text, norm, source, top_k, min_score, settings, started, query_info,
            tag_list=()):
    k = max(1, min(int(top_k or settings.top_k), settings.max_top_k))
    threshold = settings.min_score(source) if min_score is None else float(min_score)
    lex = index.lexicon
    original_norm = norm
    review_preference = index.review_preferences.get(original_norm, {})
    preferred_pose_id = review_preference.get('preferred_pose_id')
    rejected_pose_ids = review_preference.get('rejected_pose_ids', frozenset())
    if review_preference:
        query_info['review_preference'] = {
            'scope': 'exact_normalized_query', 'preferred_pose_id': preferred_pose_id,
            'rejected_pose_ids': sorted(rejected_pose_ids),
        }
    excluded_actions = set()
    contrast = re.match(r"^(.+?)(?:게|것이|거가|것은|거는)\s*아니라\s*(.+)$", norm)
    if contrast:
        left, right = contrast.groups()
        left_concepts = lex.extract(left)
        excluded_actions = {t for c in left_concepts for t in c.alternatives if t[0]=='action'}
        objects = {t for c in left_concepts for t in c.alternatives if t[0] in ('object','activity')}
        norm = ' '.join([*(lex.labels[t] for t in sorted(objects)), right])
        query_info['negated_actions'] = [':'.join(t) for t in sorted(excluded_actions)]
    # 상대의 행동은 찾는 인물의 행동이 아니다. 명시적인 원인절만 분리한다.
    # '상대에게 주먹을 날리는' 등 주체가 다른 문장은 여기서 잘라내지 않는다.
    foreground = re.match(r"^(?:상대|상대방|적)(?:가|이).+?(?:니까|면|휘두르자)(.+)$", norm)
    incoming = re.match(r"^(?:칼|공)(?:이|은).+?날아오자(.+)$", norm)
    called = re.match(r"^.+?부르니까(.+)$", norm)
    transition = re.match(r"^.+?있다가(.+)$", norm)
    if foreground or incoming or called or transition:
        event = foreground or incoming or called or transition
        # 공은 잡는 행위의 대상으로 유지하고, 날아오는 칼은 소지 도구로 읽지 않는다.
        context = ('공 ' if incoming and original_norm.startswith('공') else
                   '회피 ' if incoming and original_norm.startswith('칼') else '')
        norm = context + event.group(1).strip()
        query_info["foreground_text"] = norm
    if re.search(r'피하|피하는|회피', norm):
        norm = re.sub(r'(?:날아오는\s*)?(?:상대\s*)?(?:주먹|발차기|칼)(?:을|를)', '공격을', norm)
        query_info['foreground_text'] = norm
    norm = re.sub(r'넘어진\s+상대', '상대', norm)
    norm = re.sub(r'오랜만에\s*만난\s*', '', norm)
    surrender_context = bool(re.search(r'싸우지\s*않', norm))
    if surrender_context:
        norm = re.sub(r'^더는\s*싸우지\s*않.+?듯\s*', '항복하며 ', norm)
    if re.search(r'충격을\s*받아내', norm) and re.search(r'뛰어내|점프', norm):
        norm = re.sub(r'충격을\s*받아내\w*', '착지하는', norm)
    if norm != original_norm:
        query_info['foreground_text'] = norm
    concepts = lex.extract(_mask_stopwords(norm, lex))  # '인물이'의 '물' 같은 오매칭 방지
    # 공을 '잡으려'는 받기이고, 검을 잡으려는 소지다. 동사만 전역 catch로 바꾸지 않는다.
    ball_context = any(('object','ball') in c.alternatives for c in concepts)
    if ball_context and re.search(r'잡으려|잡는', norm):
        alts = frozenset({('action','catch')})
        surface = re.search(r'잡으려\w*|잡는',norm).group(0)
        concepts.append(Concept(surface, alts, lex.weight_of(alts), lex.miss_weight_of(alts)))
    if ball_context and re.search(r'바닥에\s*튀기', norm):
        alts = frozenset({('action','dribble')})
        surface = _nospace(re.search(r'바닥에\s*튀기\w*',norm).group(0))
        concepts.append(Concept(surface, alts, lex.weight_of(alts), lex.miss_weight_of(alts)))
    clauses = body_clauses(norm)
    spans = body_spans(norm)
    # 신체 묘사 구절 안의 동작어("몸은 앞으로 기울어")는 찾는 행동이 아니라 모양이다. 모양은
    # body 성분이 따로 재므로, 자세 태그가 아닌 개념은 못 맞춰도 덜 깎는다.
    concepts = [
        replace(c, miss_weight=min(c.penalty, settings.body_manner_miss_weight))
        if any(a <= c.word_index <= b for a, b, _ in spans)
        and not any(f == "posture" for f, _ in c.alternatives) else c
        for c in concepts
    ]
    if tag_list:
        # 확정 태그가 우선한다. 같은 뜻을 문장에서도 읽었으면 문장 쪽 개념은 뺀다.
        # 같은 필드의 값끼리는 OR이다. VLM은 'action:slash'와 'action:strike'를 함께 내는 일이
        # 흔한데, 각각을 별개 개념으로 두면 둘 다 가진 포즈만 살아남는다.
        by_field: dict[str, list[str]] = {}
        for t in dict.fromkeys(tag_list):
            by_field.setdefault(t.split(":", 1)[0], []).append(t)
        tag_concepts = []
        for field_name, values in by_field.items():
            alts = frozenset(tuple(t.split(":", 1)) for t in values)
            tag_concepts.append(Concept("tag:" + "|".join(values), alts, lex.weight_of(alts),
                                        lex.miss_weight_of(alts)))
        tagged = set().union(*(c.alternatives for c in tag_concepts))
        concepts = tag_concepts + [c for c in concepts if not (c.alternatives & tagged)]
    # 검의 내려베기는 태깅에서 slash와 strike(내리치기)로 나뉘어 있다.
    # 검 맥락에 한정해 같은 동작 후보로 읽는다(맨손/망치 공격에는 확장하지 않음).
    sword_context = any(("object", "sword") in c.alternatives for c in concepts)
    if sword_context:
        concepts = [replace(c, alternatives=c.alternatives | {("action", "strike")})
                    if ("action", "slash") in c.alternatives else c for c in concepts]
    query_info.update({
        "concepts": [c.to_dict(lex.labels) for c in concepts],
        "body_clauses": clauses,
    })
    # '농구공'의 object:ball OR activity:basketball 때문에 다른 종목이 맞는 것으로
    # 처리되지 않게 한다. 명시된 종목과 베기/던지기 같은 구체 행동은 별도 확인한다.
    requested_activities = [
        frozenset(t for t in c.alternatives if t[0] == "activity")
        for c in concepts if any(t[0] == "activity" for t in c.alternatives)
        and (("object", "ball") in c.alternatives or
             ("activity", "goalkeeping") in c.alternatives)]
    requested_actions = frozenset(t for c in concepts for t in c.alternatives
                                  if t[0] == "action" and (t[1] in DISCRETE_ACTIONS or
                                  (t[1] == "punch" and ("action", "strike") in c.alternatives)))
    floor_hand_pattern = r"(손(?:을|으로|이|은|\s).{0,18}(?<!손)바닥|(?<!손)바닥.{0,18}손)"
    floor_hand_contact = r"(손[^.]{0,18}(?<!손)바닥[^.]{0,12}(?:짚|지지|받치|대고|댄)|(?<!손)바닥[^.]{0,18}손[^.]{0,12}(?:짚|지지|받치|대고|댄))"
    requests_floor_hand = bool(re.search(floor_hand_pattern, norm)) and bool(re.search(r'짚|지지|받치|대고|댄', norm))
    requests_twohand_sword = sword_context and bool(re.search(r"양손|두\s*손", norm))
    requests_fighting = bool(re.search(r"싸우|싸움|격투", norm)) and not surrender_context
    fighting_actions = frozenset(("action", v) for v in
        ("punch", "kick", "slash", "thrust", "strike", "guard", "block", "defend", "push", "pull"))
    # 핵심 동작은 보조 행동(점프/몸 숙이기)으로 대체하지 않는다.
    action_groups = []
    for value in ('catch','dribble','evade','block','land','hug','surrender'):
        if value == 'catch' and not ball_context:
            continue  # 위협/충격을 '받아'는 공 받기가 아니다.
        if value == 'hug' and (not re.search(r'포옹|끌어안|껴안',norm) or '무릎' in norm):
            continue  # 상자를 안고 나르거나 자기 무릎을 끌어안는 것은 사람 포옹과 다르다.
        if any(('action',value) in c.alternatives for c in concepts):
            action_groups.append(frozenset({('action',value)}))
    explicit_tools = [frozenset(t for t in c.alternatives if t[0]=='object')
                      for c in concepts if any(t[0]=='object' and t[1] in
                      {'spear','bow','pistol','rifle','shield','hammer','axe','bat'} for t in c.alternatives)]
    preparation = bool(re.search(r'준비|직전|주먹.{0,15}뒤로\s*당|검.{0,15}뒤로\s*빼', norm))
    person_hug = any(('action','hug') in group for group in action_groups) and bool(re.search(r'사람|상대', norm))

    try:
        text_for_vec = lex.strip_stopwords(norm) or norm or " ".join(
            lex.labels[tuple(t.split(":", 1))] for t in tag_list)
        qvec = index.encoder.encode(text_for_vec)
        prod = index.doc_matrix @ qvec.T          # sparse(TF-IDF)·dense(임베딩) 공용
        cos = np.asarray(prod.todense() if hasattr(prod, "todense") else prod).ravel()
        if cos.shape[0] != len(index.docs):
            raise ValueError(f"encoder returned {cos.shape[0]} scores for {len(index.docs)} docs")
    except SemanticSearchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SemanticSearchError(ErrorCode.ENCODER_FAILED, f"query encoding failed: {exc}") from exc
    cos = np.nan_to_num(cos.astype(np.float64), nan=0.0)
    dense = None
    if index.dense_encoder is not None and settings.weight_dense > 0:
        try:
            dq = np.asarray(index.dense_encoder.encode(text_for_vec), dtype=np.float32).ravel()
            if dq.shape != (index.dense_matrix.shape[1],) or not np.isfinite(dq).all():
                raise ValueError("dense query vector has invalid shape or non-finite values")
            dense = index.dense_matrix @ dq
        except Exception as exc:  # noqa: BLE001
            raise SemanticSearchError(ErrorCode.ENCODER_FAILED,
                                      f"dense query encoding failed: {exc}") from exc

    total_w = sum(c.weight for c in concepts)  # (근거 표시용)
    words = query_words(norm, lex)
    word_concepts = [
        [c for c in concepts if c.surface in w or (len(w) >= 2 and w in c.surface)]
        for w, _ in words
    ]
    # 라이브러리 어디에도 없는 단어(태그 개념도 아님)는 '라이브러리 공백' 신호라 무겁게,
    # 다른 포즈에는 있는 단어는 이 포즈의 설명에 없을 뿐이라 가볍게 센다.
    known = [bool(wc) or any(_stem_in_doc(stem, d) for d in index.docs)
             for (_, stem), wc in zip(words, word_concepts)]
    unknown_ratio = (known.count(False) / len(known)) if known else 0.0
    gap_factor = (1.0 - unknown_ratio) ** settings.unknown_word_power
    query_info["content_words"] = [w for w, _ in words]
    query_info["unknown_words"] = [w for (w, _), k in zip(words, known) if not k]
    rows = []
    for i, doc in enumerate(index.docs):
        comps: dict[str, float] = {}
        matched, missing = [], []
        if concepts:
            got = denom_c = 0.0
            for c in concepts:
                hit = sorted(c.alternatives & doc.match_tags)
                if hit:
                    got += c.weight
                    denom_c += c.weight
                    for f, v in hit:
                        matched.append({
                            "surface": c.surface, "group": lex.field_group.get(f, f),
                            "field": f, "value": v, "label": lex.labels.get((f, v), v),
                        })
                else:
                    denom_c += c.penalty
                    missing.append(c.surface)
            comps["concept"] = got / denom_c if denom_c > 0 else 0.0
        body_detail = {}
        region_scores = []
        for region in ("upper", "lower"):
            if clauses[region]:
                vals = [_containment(cl, doc.desc_grams) for cl in clauses[region]]
                region_scores.append(sum(vals) / len(vals))
                body_detail[region] = round(region_scores[-1], 4)
        if region_scores:
            comps["body"] = sum(region_scores) / len(region_scores)
        comps["text"] = min(1.0, float(cos[i]) / settings.text_cos_ref)
        if dense is not None:
            # e5 코사인은 0.7~0.95에 몰린다 → [dense_lo, dense_hi]를 0~1로 편다.
            span = max(1e-6, settings.dense_hi - settings.dense_lo)
            comps["dense"] = float(min(1.0, max(0.0, (dense[i] - settings.dense_lo) / span)))
        weights = {"concept": settings.weight_concept, "body": settings.weight_body,
                   "text": settings.weight_text, "dense": settings.weight_dense}
        denom = sum(weights[n] for n in comps)
        base = sum(weights[n] * comps[n] for n in comps) / denom if denom else 0.0
        if set(comps) <= {"text", "dense"}:
            base *= settings.text_only_factor  # 어휘 근거 없이 문장 유사도만 남은 경우
        covered = [
            any(c.alternatives & doc.match_tags for c in wc) or _stem_in_doc(stem, doc)
            for (_, stem), wc in zip(words, word_concepts)
        ]
        known_cov = [ok for ok, k in zip(covered, known) if k]
        coverage = sum(known_cov) / len(known_cov) if known_cov else 1.0
        comps["coverage"] = coverage
        if gap_factor < 1.0:
            comps["gap_factor"] = gap_factor
        score = base * (settings.coverage_floor + (1.0 - settings.coverage_floor) * coverage) \
            * gap_factor
        required_match = (
            (not requested_actions or bool(requested_actions & doc.match_tags)) and
            all(a & doc.match_tags for a in requested_activities) and
            (not requests_floor_hand or bool(re.search(floor_hand_contact, normalize_text(doc.description)))) and
            (not sword_context or ("object", "sword") in doc.match_tags) and
            (not requests_twohand_sword or bool(re.search(r"양손|두\s*손", doc.description))) and
            (not requests_fighting or bool(fighting_actions & doc.match_tags)) and
            all(group & doc.match_tags for group in action_groups) and
            all(group & doc.match_tags for group in explicit_tools) and
            not bool(excluded_actions & doc.tags) and
            (not person_hug or not re.search(r'꽃다발|상자|곰인형|무릎',doc.description))
        )
        failed_relevance = relevance_constraints(norm, normalize_text(doc.description), doc.match_tags)
        if failed_relevance or doc.pose_id in rejected_pose_ids:
            required_match = False
            missing.extend('requirement:' + reason for reason in failed_relevance)
            if doc.pose_id in rejected_pose_ids:
                missing.append('explicit_human_rejection')
        if not required_match:
            # 틀린 종목/행동을 가까운 글자·몸 묘사로 만회해 Top-K를 채우지 않는다.
            score *= 0.25
            comps["requested_constraint_match"] = 0.0
        else:
            comps["requested_constraint_match"] = 1.0
        if preparation and not (('action','prepare') in doc.tags or ('intent','prepare_action') in doc.tags
                                or re.search(r'준비|직전|뒤로.*당', doc.description)):
            score *= .65
            comps['preparation_match'] = 0.0
        uncovered = [w for (w, _), ok in zip(words, covered) if not ok]
        rows.append((score, i, comps, matched, missing, body_detail, float(cos[i]), uncovered))

    rows.sort(key=lambda r: (-r[0], index.docs[r[1]].pose_id))
    # Feedback cannot resurrect low-score, constraint-failing, missing or excluded assets.
    # Retrieval scores remain unchanged; this is an explicitly labeled second-stage order.
    if preferred_pose_id:
        for row in rows:
            if (index.docs[row[1]].pose_id == preferred_pose_id and row[0] >= threshold
                    and row[2].get('requested_constraint_match') == 1.0):
                row[2]['human_preference'] = 1.0
                rows.remove(row)
                rows.insert(0, row)
                break

    picked, alternates, seen_family = [], {}, {}
    for row in rows:
        pose_id = index.docs[row[1]].pose_id
        fam = index.family.get(pose_id, pose_id) if settings.collapse_mirrors else pose_id
        if fam in seen_family:
            alternates.setdefault(seen_family[fam], []).append(
                {"pose_id": pose_id, "score": round(row[0], 4)})
            continue
        seen_family[fam] = pose_id
        picked.append(row)
        if len(picked) >= k + settings.near_miss_count:
            break

    above = [r for r in picked if r[0] >= threshold][:k]
    candidates = []
    for rank, (score, i, comps, matched, missing, body_detail, raw_cos, uncovered) in enumerate(above, 1):
        pose_id = index.docs[i].pose_id
        pose = index.pose_json(pose_id)
        if pose is None:
            raise SemanticSearchError(ErrorCode.POSE_JSON_MISSING,
                                      f"pose JSON could not be built for {pose_id}",
                                      {"pose_id": pose_id})
        if pose.get("pose_id") != pose_id:
            raise SemanticSearchError(ErrorCode.POSE_ID_MISMATCH,
                                      f"candidate {pose_id} != pose JSON {pose.get('pose_id')}",
                                      {"candidate_pose_id": pose_id,
                                       "pose_json_pose_id": pose.get("pose_id")})
        candidates.append({
            "rank": rank,
            "pose_id": pose_id,
            "score": round(score, 4),
            "score_components": {n: round(v, 4) for n, v in comps.items()},
            "evidence": {
                "matched_tags": matched,
                "missing_concepts": missing,
                "uncovered_words": uncovered,
                "body": body_detail,
                "text_cosine": round(raw_cos, 4),
                "description": index.docs[i].description,
            },
            "mirror_alternates": alternates.get(pose_id, []),
            "refine_allowed": False,
            "pose": pose,
        })
    near = [] if candidates else [
        {"pose_id": index.docs[r[1]].pose_id, "score": round(r[0], 4)}
        for r in picked[: settings.near_miss_count]
    ]
    return {
        "status": STATUS_OK if candidates else STATUS_NO_RESULT,
        "error": None,
        "query": query_info,
        "candidates": candidates,
        "near_misses": near,
        "threshold": threshold,
        "top_k": k,
        "index": index.identity,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


# ─────────────────────────────── 서비스 ───────────────────────────────


class SemanticSearchService:
    """기동 시 1회 색인을 올리고, 실패해도 예외 대신 error 상태로 답하는 래퍼.

    직접 검색 HTTP 핸들러가 사용한다. 동적 라우팅 연결은 후속 작업이다::

        service = SemanticSearchService.from_config(CFG, db_path=..., allowed_pose_ids=...)
        result = service.search("공을 던지는 투수", source="user")
        if result["status"] == "ok": ...
    """

    def __init__(self, index: Optional[SemanticIndex], settings: SemanticSettings,
                 load_error: Optional[SemanticSearchError] = None):
        self.index = index
        self.settings = settings
        self.load_error = load_error
        # 기동 이후 누적 — /healthz에 싣는다. no_result 비율이 라이브러리 공백 지표다.
        self._stats: Counter = Counter()
        import threading
        self._lock = threading.Lock()

    def _record(self, result: dict, source: str) -> None:
        with self._lock:
            self._stats[f"status.{result['status']}"] += 1
            self._stats[f"source.{source}"] += 1
            if result["status"] == STATUS_ERROR:
                self._stats[f"error.{result['error']['code']}"] += 1

    def stats(self) -> dict:
        with self._lock:
            return dict(sorted(self._stats.items()))

    @classmethod
    def from_paths(cls, settings: SemanticSettings, **load_kwargs) -> "SemanticSearchService":
        settings.validate()
        try:
            return cls(SemanticIndex.load(**load_kwargs), settings)
        except SemanticSearchError as exc:
            return cls(None, settings, exc)
        except Exception as exc:  # noqa: BLE001
            return cls(None, settings,
                       SemanticSearchError(ErrorCode.INDEX_LOAD_FAILED, f"{type(exc).__name__}: {exc}"))

    @classmethod
    def from_config(cls, cfg, *, db_path: str, allowed_pose_ids=None,
                    library_version: Optional[str] = None) -> "SemanticSearchService":
        settings = settings_from_config(cfg)
        if not cfg.semantic_enabled:
            return cls(None, settings, SemanticSearchError(
                ErrorCode.INDEX_NOT_LOADED, "semantic search disabled (SEMANTIC_ENABLED=0)"))
        dense = None
        if cfg.semantic_dense_enabled:
            from .semantic_embedding import (EmbeddingUnavailable, load_dense_encoder,
                                             model_dir_default)
            try:
                dense = load_dense_encoder(cfg.semantic_embedding_dir
                                           or model_dir_default(cfg.data_dir))
            except EmbeddingUnavailable as exc:
                return cls(None, settings, SemanticSearchError(
                    ErrorCode.ENCODER_FAILED, f"dense encoder unavailable: {exc}"))
        return cls.from_paths(
            settings,
            dense_encoder=dense,
            tags_path=cfg.semantic_tags_path or os.path.join(cfg.data_dir, "semantic", "search-v2.json"),
            taxonomy_path=cfg.semantic_taxonomy_path
            or os.path.join(cfg.data_dir, "semantic", "search-taxonomy-v2.json"),
            lexicon_path=cfg.semantic_lexicon_path,
            db_path=db_path,
            allowed_pose_ids=allowed_pose_ids,
            verify_db_sha256=cfg.semantic_verify_db_sha256,
            thumbs_dir=os.path.join(cfg.data_dir, "thumbs"),
            library_version=library_version,
            review_preferences_path=getattr(cfg, 'semantic_review_preferences_path', '') or None,
        )

    @property
    def ready(self) -> bool:
        return self.index is not None

    def health(self) -> dict:
        if self.index is not None:
            return {"ready": True, **self.index.identity, "stats": self.stats()}
        err = self.load_error
        return {"ready": False, "error_code": err.code if err else ErrorCode.INDEX_NOT_LOADED,
                "error": err.message if err else "not loaded", "stats": self.stats()}

    def search(self, text: str, *, source: str = "user", top_k: Optional[int] = None,
               min_score: Optional[float] = None,
               tags: Optional[Sequence[str]] = None) -> dict:
        result = self._search(text, source=source, top_k=top_k, min_score=min_score, tags=tags)
        self._record(result, source if source in SOURCES else "invalid")
        return result

    def _search(self, text, *, source, top_k, min_score, tags) -> dict:
        if self.index is None:
            err = self.load_error or SemanticSearchError(ErrorCode.INDEX_NOT_LOADED, "not loaded")
            started = time.perf_counter()
            result = _error(err.code, err.message, detail=err.detail, started=started,
                            query={"text": text, "source": source,
                                   "normalized": normalize_text(text if isinstance(text, str) else "")})
            # 쿼리 자체가 잘못된 경우는 색인 상태보다 먼저 알려 준다.
            pre = semantic_search(None, text, source=source, settings=self.settings, tags=tags)
            if pre["error"]["code"] in (ErrorCode.INVALID_QUERY, ErrorCode.INVALID_SOURCE,
                                        ErrorCode.INVALID_TAGS):
                return pre
            return result
        return semantic_search(self.index, text, source=source, top_k=top_k,
                               min_score=min_score, settings=self.settings, tags=tags)


def settings_from_config(cfg) -> SemanticSettings:
    return SemanticSettings(
        top_k=cfg.semantic_top_k,
        max_top_k=cfg.semantic_max_top_k,
        min_score_user=cfg.semantic_min_score_user,
        min_score_vlm=cfg.semantic_min_score_vlm,
        weight_concept=cfg.semantic_weight_concept,
        weight_body=cfg.semantic_weight_body,
        weight_text=cfg.semantic_weight_text,
        text_cos_ref=cfg.semantic_text_cos_ref,
        coverage_floor=cfg.semantic_coverage_floor,
        unknown_word_power=cfg.semantic_unknown_word_power,
        body_manner_miss_weight=cfg.semantic_body_manner_miss_weight,
        text_only_factor=cfg.semantic_text_only_factor,
        weight_dense=cfg.semantic_weight_dense if cfg.semantic_dense_enabled else 0.0,
        dense_lo=cfg.semantic_dense_lo,
        dense_hi=cfg.semantic_dense_hi,
        collapse_mirrors=cfg.semantic_collapse_mirrors,
    )
