"""Offline contact-part recipes. Intent input is explicit, never inferred from reviews.

These procedural recipes are distinct from v1 angle-only BVH parts. Their target
frames attach to the destination anatomy; no case IDs or world coordinates exist
in this module. Nothing here changes production search/refine contracts.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ContactPart:
    id: str
    label: str
    target: str
    patch: str
    hand_preset: str
    elbow_rule: str
    aliases: tuple[str, ...]
    planar_thumb: bool = False
    lateral_fraction: float = 0
    max_patch_median_m: float = .008
    max_patch_p90_m: float = .015
    max_normal_error_deg: float = 25
    max_wrist_axis_deg: float = 65

    def record(self):
        return {"schema": "standin.arm-contact-part.v1", **asdict(self),
                "authoring": "procedural_anatomical_contact_recipe",
                "coordinate_frame": "destination_target_anatomy",
                "wrist_policy": "align_contact_frame", "fingers_policy": "explicit_selected_hand_preset",
                "paired": False, "release_allowed": False}


PARTS = (
    ContactPart("waist_side", "엄지로 허리 옆 걸치기", "waist_side", "thumb_pad", "relaxed", "outward_low",
                ("허리에 손", "허리 옆 짚기", "한 손 허리")),
    ContactPart("head_side", "머리 옆 짚기", "head_side", "palm", "open", "forward_below_wrist",
                ("머리 옆 짚기", "머리 옆에 손", "관자놀이 짚기"), planar_thumb=True),
    ContactPart("chin_support", "주먹으로 턱 받치기", "chin_underside", "fist_top", "fist", "low_below_wrist",
                ("턱 받침", "턱 받치기", "주먹으로 턱 받치기")),
    ContactPart("chin_relaxed", "느슨한 손으로 턱 받치기", "chin_underside", "thumb_pad", "rest_chin", "low_below_wrist",
                ("느슨한 손 턱 받침",)),
    ContactPart("neck_back", "뒷목에 손", "neck_back", "palm", "open", "outward_raised",
                ("뒷목 잡기", "목 뒤에 손"), planar_thumb=True),
    ContactPart("abdomen_rest", "배 위에 손 얹기", "abdomen_front", "palm", "open", "side_below_shoulder",
                ("배 위에 손", "아랫배에 손 얹기"), planar_thumb=True),
    ContactPart("chest_touch", "가슴 앞 손", "chest_front", "palm", "open", "side_below_shoulder",
                ("가슴 앞에 손", "가슴에 손 얹기"), planar_thumb=True),
    ContactPart("collar_touch", "목 아래 옷깃 잡기", "collar_front", "thumb_pad", "clutch_chest", "side_below_shoulder",
                ("옷깃 잡기", "목 아래 가슴 짚기")),
    ContactPart("forehead_touch", "이마 짚기", "forehead_front", "palm", "open", "outward_raised",
                ("이마에 손", "이마 짚기"), planar_thumb=True),
    ContactPart("knee_rest", "세운 무릎 위 주먹 얹기", "same_knee", "ulnar_edge", "fist", "side_below_shoulder",
                ("무릎 위에 주먹",)),
)


def select_contact_part(intent, side):
    if side not in ("left", "right"):
        raise ValueError("contact part requires an explicit anatomical side")
    query = " ".join(intent.split())
    if "팔짱" in query or "fold" in query or "양손" in query:
        raise ValueError("paired contact requires a validated atomic two-arm recipe; held")
    matches = [p for p in PARTS if query == p.id or query in p.aliases]
    if len(matches) != 1:
        raise ValueError("no exact contact intent; different anatomical targets are not interchangeable")
    return matches[0]


def contact_axis(part):
    return {"palm":1,"thumb_pad":1,"fist_top":0,"ulnar_edge":2}[part.patch]


def admission_reasons(metrics, part):
    """Missing measurements fail closed; no score can offset a failed condition."""
    limits = {
        "patch_median_m": part.max_patch_median_m,
        "patch_p90_m": part.max_patch_p90_m,
        "normal_error_deg": part.max_normal_error_deg,
        "hand_frame_error_deg": .1,
        "wrist_axis_deg": part.max_wrist_axis_deg,
        "elbow_flexion_deg": 165,
        "wrist_swing_deg": 65,
        "wrist_twist_deg": 35,
        "bone_length_error_m": 2e-5,
        "wrist_goal_error_m": 2e-5,
        "protected_matrix_delta": 2e-5,
    }
    reasons = [key for key, maximum in limits.items()
               if not isinstance(metrics.get(key), (int, float))
               or not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= maximum]
    if not isinstance(metrics.get("elbow_rule_margin_m"), (int, float)) or not math.isfinite(metrics["elbow_rule_margin_m"]) or metrics["elbow_rule_margin_m"] < 0:
        reasons.append("elbow_rule")
    if metrics.get("new_skin") != []:
        reasons.append("new_skin")
    if metrics.get("patch_samples", 0) < 8:
        reasons.append("missing_patch")
    return reasons
