"""Visibility must not depend on target error or a neighboring bone's length."""
import numpy as np
import pytest

from src import refine
from src.features import _BONES
from src.refine_projection import FixedProjection, projection_scope


class Probe:
    axis = 0
    length = 1.
    param_idx = np.array([0])
    base_frame = np.array([0.])

    def __init__(self, *args):
        pass

    def joints3d(self, params):
        points = np.zeros((17, 3))
        points[[5, 6], 1] = 1.
        points[7, self.axis] += float(params[0]) * self.length / 100
        points[9, self.axis] += float(params[0]) * self.length / 50
        return points, np.ones(17)


def observe(target=1., valid=True):
    return refine.limb_observability([], [], "front", "left_arm",
        np.full((len(_BONES), 2), target), np.full(len(_BONES), valid),
        np.ones(len(_BONES)))


def test_visible_motion_independent_of_target_and_bone_length(monkeypatch):
    monkeypatch.setattr(refine, "_Forward", Probe)
    monkeypatch.setattr(Probe, "axis", 0)
    for length in (.01, 1., 100.):
        monkeypatch.setattr(Probe, "length", length)
        assert observe(1.) == pytest.approx(1., abs=1e-6)
        assert observe(-1.) == pytest.approx(observe(1.), abs=1e-9)
        assert observe(0.) == pytest.approx(observe(1.), abs=1e-9)


def test_depth_only_motion_and_missing_targets_remain_closed(monkeypatch):
    monkeypatch.setattr(refine, "_Forward", Probe)
    monkeypatch.setattr(Probe, "axis", 2)
    assert observe() == 0.
    monkeypatch.setattr(Probe, "axis", 0)
    assert observe(valid=False) == 0.


def test_fixed_perspective_uses_baseline_depth_units(monkeypatch):
    monkeypatch.setattr(refine, "_Forward", Probe)
    monkeypatch.setattr(Probe, "axis", 0)
    for distance in (2., 20.):
        with projection_scope(FixedProjection((0., 0., 0.), distance, .1)):
            assert observe() == pytest.approx(1., abs=1e-6)
