"""Single source of review thresholds; these are triage guards, not clinical ROM."""

import hashlib
import json

VERSION = "2026-10-02.1"
VIEWS = ("front", "three_quarter", "side", "back")
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
WRIST_BEND = 65
LIMB_BEND = 155
ELBOW_HINGE = 65
SPINE_ROTATION = 35
CAPSULE_DEPTH = 0.025
CHARACTER_SHA256 = "7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d"
VISUAL_CHECKS = {
    "four_views": "정면·45°·측면·후면을 모두 확인",
    "joints_torso": "팔꿈치·무릎 방향과 골반·허리의 꺾임·비틀림 확인",
    "contacts_balance": "몸 관통, 손·발 방향, 접지 또는 공중 동작의 균형 확인",
    "hands": "양손 손가락과 의도한 손 모양 확인",
    "action_source": "동작 실루엣·중복·출처·합성 여부 확인",
}


def manifest():
    return {
        "version": VERSION,
        "views": VIEWS,
        "finger_joints": 30,
        "wrist_bend_degrees": WRIST_BEND,
        "limb_bend_degrees": LIMB_BEND,
        "elbow_hinge_degrees": ELBOW_HINGE,
        "spine_rotation_degrees": SPINE_ROTATION,
        "capsule_depth_torso_units": CAPSULE_DEPTH,
        "character_sha256": CHARACTER_SHA256,
        "visual_checks": VISUAL_CHECKS,
    }


def fingerprint():
    return hashlib.sha256(json.dumps(manifest(), sort_keys=True).encode()).hexdigest()
