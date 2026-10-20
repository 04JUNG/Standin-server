"""응답에 실리는 추출 모델 버전(`pose_model_version`).

라이브러리 버전과 같은 역할을 모델 쪽에서 한다 — 어느 모델이 답했는지 모르면 모델을
바꾼 전후를 비교할 수 없다.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import STATE, _pose_model_version
from src.config import CFG


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("POSE_MODEL_VERSION", raising=False)
    STATE.pop("pose_model_bundle", None)
    yield
    STATE.pop("pose_model_bundle", None)


def test_bundle_identity_is_used_when_present():
    STATE["pose_model_bundle"] = {"model_id": "humanart-m", "build_id": "20261002-1"}
    assert _pose_model_version() == "humanart-m@20261002-1"


def test_model_id_alone_still_beats_a_constant():
    STATE["pose_model_bundle"] = {"model_id": "humanart-m"}
    assert _pose_model_version() == "humanart-m"


def test_without_a_bundle_the_variant_is_recorded(monkeypatch):
    # 예전에는 여기서 "runtime-default"가 나갔다. 그 값으로는 아무것도 되짚을 수 없다.
    monkeypatch.setattr(CFG, "pose_model_variant", "cascade")
    assert _pose_model_version() == "cascade"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("POSE_MODEL_VERSION", "pinned-for-an-experiment")
    STATE["pose_model_bundle"] = {"model_id": "humanart-m", "build_id": "20261002-1"}
    assert _pose_model_version() == "pinned-for-an-experiment"


def test_the_old_constant_is_gone(monkeypatch):
    monkeypatch.setattr(CFG, "pose_model_variant", "current-x")
    assert _pose_model_version() != "runtime-default"
