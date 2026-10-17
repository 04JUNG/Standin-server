"""Dimensionless measurements of an authored neutral mesh, never VLM coordinates."""
from __future__ import annotations
import numpy as np

VERSION = "body-morphology.v1"

def measure_neutral(vertices, landmarks, joints):
    co = np.asarray(vertices, dtype=float)
    if co.ndim != 2 or co.shape[1] != 3 or len(co) < 20 or not np.isfinite(co).all():
        raise ValueError("invalid_neutral_mesh")
    lm = {k: float(landmarks[k]) for k in ("chin", "hip", "shoulder", "knee", "ankle")}
    floor, top = float(co[:, 2].min()), float(co[:, 2].max())
    if not all(np.isfinite(list(lm.values()))) or not floor <= lm["ankle"] < lm["knee"] < lm["hip"] < lm["shoulder"] < lm["chin"] < top:
        raise ValueError("invalid_landmark_order")
    height = top-floor
    def joint(name):
        p = np.asarray(joints[name], dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all(): raise ValueError("invalid_joint")
        return p
    def distance(a, b): return float(np.linalg.norm(joint(a)-joint(b)))
    def section(z, positive=False):
        # Fixed scale-relative bands, robust to tiny tessellation changes.
        band = co[np.abs(co[:, 2]-z) <= height*.008]
        if positive: band = band[band[:, 0] > height*.012]
        if len(band) < 4: raise ValueError("missing_measurement_section")
        return np.ptp(band[:, 0]), np.ptp(band[:, 1])
    torso = lm["shoulder"]-lm["hip"]
    chest_w, chest_d = section(lm["hip"]+.65*torso)
    waist_w, waist_d = section(lm["hip"]+.25*torso)
    hip_w, hip_d = section(lm["hip"])
    thigh_w, thigh_d = section(lm["knee"]+.65*(lm["hip"]-lm["knee"]), True)
    calf_w, calf_d = section(lm["ankle"]+.55*(lm["knee"]-lm["ankle"]), True)
    shoulder = distance("LeftArm", "RightArm")
    arm = distance("LeftArm", "LeftForeArm")+distance("LeftForeArm", "LeftHand")
    leg = distance("LeftUpLeg", "LeftLeg")+distance("LeftLeg", "LeftFoot")
    midpoint = (joint("LeftArm")+joint("LeftForeArm"))/2
    band = co[np.abs(co[:, 0]-midpoint[0]) <= height*.008]
    if len(band) < 4: raise ValueError("missing_arm_section")
    arm_z, arm_d = np.ptp(band[:, 2]), np.ptp(band[:, 1])
    out = dict(head_units=height/(top-lm["chin"]), torso_height=torso/height,
        leg_torso=leg/torso, arm_torso=arm/torso, shoulder_hip=shoulder/hip_w,
        shoulder_height=shoulder/height, chest_waist=chest_w/waist_w,
        waist_hip=waist_w/hip_w, chest_depth_height=chest_d/height,
        abdomen_depth_height=waist_d/height, hip_depth_height=hip_d/height,
        upper_arm_height=arm_z/height, upper_arm_depth_height=arm_d/height,
        thigh_width_height=thigh_w/height, thigh_depth_height=thigh_d/height,
        calf_width_height=calf_w/height, calf_depth_height=calf_d/height)
    if not all(np.isfinite(v) and v > 0 for v in out.values()): raise ValueError("invalid_morphology_ratio")
    return {k: round(float(v), 6) for k, v in out.items()}
