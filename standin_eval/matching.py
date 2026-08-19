from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import bbox


@dataclass(frozen=True)
class MatchPolicy:
    version: int = 1
    min_iou: float = 0.10
    max_center_distance: float = 0.75

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "min_iou": self.min_iou,
            "max_center_distance": self.max_center_distance,
        }


def intersection_over_union(first, second) -> float:
    ax1, ay1, ax2, ay2 = bbox(list(first), "first_bbox")
    bx1, by1, bx2, by2 = bbox(list(second), "second_bbox")
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_distance(gt, prediction) -> float:
    gx1, gy1, gx2, gy2 = bbox(list(gt), "gt_bbox")
    px1, py1, px2, py2 = bbox(list(prediction), "prediction_bbox")
    gcx, gcy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    width = max(gx2 - gx1, 1e-9)
    height = max(gy2 - gy1, 1e-9)
    return math.sqrt(((pcx - gcx) / width) ** 2 + ((pcy - gcy) / height) ** 2)


def _hungarian(cost: list[list[float]]) -> list[int]:
    """Return column per row for a square/rectangular minimum-cost matrix."""
    if not cost:
        return []
    rows, columns = len(cost), len(cost[0])
    if rows > columns:
        transposed = [[cost[row][column] for row in range(rows)] for column in range(columns)]
        inverse = _hungarian(transposed)
        result = [-1] * rows
        for column, row in enumerate(inverse):
            if row >= 0:
                result[row] = column
        return result

    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, columns + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(columns + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * rows
    for j in range(1, columns + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def match_people(
    cut_id: str,
    gt_people: list[dict],
    predictions: list[dict],
    route: str,
    expected_route: str,
    policy: MatchPolicy | None = None,
) -> list[dict]:
    policy = policy or MatchPolicy()
    real_predictions = [row for row in predictions if row.get("box_xyxy") is not None]
    size = max(len(gt_people), len(real_predictions))
    if size == 0:
        return []
    costs = [[1.0 for _ in range(size)] for _ in range(size)]
    details: dict[tuple[int, int], tuple[float, float]] = {}
    for i, person in enumerate(gt_people):
        for j, prediction in enumerate(real_predictions):
            overlap = intersection_over_union(person["bbox_xyxy"], prediction["box_xyxy"])
            center = normalized_center_distance(person["bbox_xyxy"], prediction["box_xyxy"])
            center_score = max(0.0, 1.0 - min(center, 1.0))
            score = max(overlap, 0.25 * center_score)
            costs[i][j] = 1.0 - score + i * 1e-12 + j * 1e-14
            details[(i, j)] = (overlap, center)
    assignment = _hungarian(costs)

    rows: list[dict] = []
    matched_prediction_ids: set[str] = set()
    for i, person in enumerate(gt_people):
        column = assignment[i] if i < len(assignment) else -1
        prediction = real_predictions[column] if 0 <= column < len(real_predictions) else None
        overlap, center = details.get((i, column), (0.0, float("inf")))
        accepted = bool(
            prediction is not None
            and (overlap >= policy.min_iou or center <= policy.max_center_distance)
        )
        if accepted:
            matched_prediction_ids.add(prediction["prediction_id"])
            rows.append({
                "cut_id": cut_id,
                "person_id": person["person_id"],
                "prediction_id": prediction["prediction_id"],
                "match_status": "matched",
                "iou": overlap,
                "normalized_center_distance": center,
                "match_policy_version": policy.version,
            })
        else:
            if expected_route == "core" and route != "core":
                reason = "wrong_route"
            elif not real_predictions:
                reason = "no_prediction"
            else:
                reason = "below_match_threshold"
            rows.append({
                "cut_id": cut_id,
                "person_id": person["person_id"],
                "prediction_id": None,
                "match_status": "missed",
                "miss_reason": reason,
                "iou": overlap if prediction is not None else None,
                "normalized_center_distance": center if prediction is not None else None,
                "match_policy_version": policy.version,
            })

    for prediction in predictions:
        if prediction["prediction_id"] not in matched_prediction_ids:
            rows.append({
                "cut_id": cut_id,
                "person_id": None,
                "prediction_id": prediction["prediction_id"],
                "match_status": "false_positive",
                "match_policy_version": policy.version,
            })
    return rows
