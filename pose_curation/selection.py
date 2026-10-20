"""Deterministic candidate selection, independent of download and export."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .motion import normalized_body


@dataclass(frozen=True)
class Selection:
    sample_index: int
    reason: str
    speed: float
    novelty: float


def select_frames(points: np.ndarray, times: np.ndarray, *, count: int,
                  minimum_distance: float, minimum_seconds: float) -> list[Selection]:
    """Combine held poses, dynamic extrema and phase coverage, then diversify.

    Distances are RMS body-joint distance in torso units. Speed uses seconds,
    not frame indices. Sampling is fixed upstream, making FPS changes comparable.
    """
    if count < 1 or len(points) < 2 or len(points) != len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("selection requires increasing timestamps and at least two poses")
    body = normalized_body(points)
    speed = np.zeros(len(body))
    speed[1:] = np.sqrt(np.mean(np.sum(np.diff(body, axis=0) ** 2, axis=-1), axis=-1)) / np.diff(times)
    speed[0] = speed[1]
    # Smooth only the selection metric. Never modify the original BVH channels.
    speed = np.convolve(np.pad(speed, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid")
    median = np.median(body, axis=0)
    deviation = np.sqrt(np.mean(np.sum((body - median) ** 2, axis=-1), axis=-1))
    candidates: dict[int, str] = {}
    for index in np.linspace(0, len(body) - 1, min(12, len(body)), dtype=int):
        candidates[int(index)] = "phase_coverage"
    for index in range(1, len(body) - 1):
        if speed[index] <= min(speed[index - 1], speed[index + 1]):
            candidates[index] = "held_pose"
        if deviation[index] >= max(deviation[index - 1], deviation[index + 1]):
            candidates[index] = "expressive_extreme"
    # Extremes in limb reach and body level include motion with no low-speed hold.
    for joint in (2, 3, 4, 5, 8, 9, 10, 11):
        for axis in (0, 1, 2):
            for index in (np.argmin(body[:, joint, axis]), np.argmax(body[:, joint, axis])):
                candidates[int(index)] = "limb_extreme"
    pool = sorted(candidates)
    first = max(pool, key=lambda i: (float(deviation[i]), -i))
    chosen = [Selection(first, candidates[first], float(speed[first]), float(deviation[first]))]
    while len(chosen) < count:
        eligible = [i for i in pool if all(abs(times[i] - times[c.sample_index]) >= minimum_seconds for c in chosen)]
        if not eligible:
            break
        distances = np.stack([np.sqrt(np.mean(np.sum((body[eligible] - body[c.sample_index]) ** 2, axis=-1), axis=-1)) for c in chosen])
        nearest = distances.min(axis=0)
        best = int(np.argmax(nearest))
        if nearest[best] < minimum_distance:
            break
        index = eligible[best]
        chosen.append(Selection(index, candidates[index], float(speed[index]), float(nearest[best])))
    return sorted(chosen, key=lambda candidate: candidate.sample_index)
