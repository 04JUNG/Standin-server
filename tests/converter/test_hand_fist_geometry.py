"""Actual registered hand proportions, not only the idealized gallery hand."""

import json
from pathlib import Path

import numpy as np
import pytest

from converter.hand_collision import contacts
from converter.hand_geometry import FingerRest, finger_rotations, hand_frame
from converter.hand_library import (
    FINGERS,
    LIBRARY_PATH,
    HandLibraryError,
    load_library,
    parse_library,
)

RIGS = json.loads((Path(__file__).parent / "fixtures/hand_fist_rigs.json").read_text())[
    "rigs"
]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.update(type="unknown"),
        lambda a: a.update(thumb_directions=[[0, 0, 0]] * 3),
        lambda a: a.update(thumb_directions=[[0, 1, 0]]),
        lambda a: a["thumb_directions"][0].__setitem__(0, float("nan")),
        lambda a: a["fan"].pop("pinky"),
        lambda a: a["fan"].update(index=True),
        lambda a: a.update(extra=True),
        lambda a: a.update(smoothing_iterations=True),
        lambda a: a.update(smoothing_iterations=-1),
        lambda a: a.update(smoothing_iterations=13),
    ],
)
def test_invalid_articulation_is_rejected_before_blender(mutate):
    payload = json.loads(LIBRARY_PATH.read_text())
    mutate(payload["presets"]["fist"]["articulation"])
    with pytest.raises(HandLibraryError):
        parse_library(json.dumps(payload).encode())


def geometry(rig, rotation=np.eye(3), offset=np.zeros(3)):
    g = rig["geometry"]
    wrist = rotation @ np.array(g["wrist"]) + offset
    bases = {f: rotation @ np.array(v) + offset for f, v in g["bases"].items()}
    rests = {}
    for key, rest in g["rests"].items():
        finger, joint = key.split(":")
        rests[finger, int(joint)] = FingerRest(
            rotation @ np.array(rest["head"]) + offset,
            rotation @ np.array(rest["direction"]),
            rotation @ np.array(rest["basis"]),
        )
    return wrist, bases, rests


def chains(rig, rests, rotations):
    result = {}
    for f in FINGERS:
        delta = np.eye(3)
        head = rests[f, 0].head.copy()
        points = [head.copy()]
        for j in range(3):
            rest = rests[f, j]
            if j:
                head = points[-2] + delta @ (rest.head - rests[f, j - 1].head)
                points[-1] = head.copy()
            delta = delta @ rest.basis @ rotations[f, j] @ rest.basis.T
            raw = rig["geometry"]["rests"][f"{f}:{j}"]
            length = np.linalg.norm(np.array(raw["tail"]) - raw["head"])
            points.append(head + delta @ rest.direction * length)
        result[f] = points
    return result


@pytest.mark.parametrize("rig", RIGS, ids=lambda r: r["model"] + "-" + r["side"])
@pytest.mark.parametrize("preset_id", ["fist", "tremble_fist"])
def test_closed_fist_is_clear_on_actual_registered_hand_proportions(rig, preset_id):
    wrist, bases, rests = geometry(rig)
    frame = hand_frame(rig["side"], wrist, bases)
    rotations = finger_rotations(
        rig["side"], load_library().preset(preset_id), frame, rests, bases
    )
    found = contacts(
        chains(rig, rests, rotations),
        palm_normal=frame.palm_normal,
        pinky_side=frame.pinky_side,
        length_axis=frame.length_axis,
        wrist=wrist,
        hand_length=rig["geometry"]["length"],
    )
    assert not found, [(c.a, c.b, c.depth) for c in found]
    for matrix in rotations.values():
        assert np.isfinite(matrix).all()
        assert np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-5)
        assert np.linalg.det(matrix) > 0.9999


@pytest.mark.parametrize("rig", RIGS, ids=lambda r: r["model"] + "-" + r["side"])
def test_fist_uses_hand_frame_after_body_pose_is_baked_into_rest(rig):
    wrist, bases, rests = geometry(rig)
    side = rig["side"]
    preset = load_library().preset("fist")
    before = finger_rotations(
        side, preset, hand_frame(side, wrist, bases), rests, bases
    )
    rng = np.random.default_rng(36)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    wrist, bases, rests = geometry(rig, rotation, np.array([2.0, -3.0, 1.0]))
    after = finger_rotations(side, preset, hand_frame(side, wrist, bases), rests, bases)
    for key in before:
        assert np.allclose(before[key], after[key], atol=2e-7)
