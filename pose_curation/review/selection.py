"""One revision-bound exclusion rule shared by review and library export."""

from .catalog import Pose


def decision(pose: Pose, reviews: dict) -> dict:
    current = reviews.get((pose.key, pose.content_hash))
    prior = [row for (key, _), row in reviews.items() if key == pose.key]
    latest = max(prior, key=lambda row: row.get("updated_at", ""), default={})
    if latest.get("status") == "rejected" and (
        current is None or current.get("updated_at", "") < latest.get("updated_at", "")
    ):
        return {**latest, "inherited_exclusion": True}
    if current is not None:
        return current
    return {"status": "pending", "note": ""}


def is_excluded(pose: Pose, reviews: dict) -> bool:
    return decision(pose, reviews)["status"] == "rejected"


def is_publishable(pose: Pose, reviews: dict) -> bool:
    status = decision(pose, reviews)["status"]
    return status != "rejected" if pose.group == "existing" else status == "accepted"
