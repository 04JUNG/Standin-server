"""Run with Blender --background --python this_file -- native-source.fbx new-output rig-profile.json.

Actual native FBX matrices exercise ordering and complete transaction rollback.
Artificial 5-degree perturbations are regression fixtures, never score evidence.
"""
import json
import sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
source, out, profile_file = map(Path, sys.argv[sys.argv.index("--")+1:])
from converter import retarget as rt
from src.refine_v4.skin_collision import inspect_self
from src.refine_v4.pipeline import Stage, run_refine_v4, snapshot, skin_nonregression
from src.refine_v4.leg_adapter import plan_legs, extract_native_leg_part, apply_native_leg_part

rt.reset_scene(); arm, meshes = rt.import_character(str(source))
b = {p.name.split(":")[-1]:p for p in arm.pose.bones}
bases = {p.name:p.matrix_basis.copy() for p in arm.pose.bones}
initial = snapshot(arm)
rows = []


def reset():
    for p in arm.pose.bones: p.matrix_basis = bases[p.name]
    bpy.context.view_layer.update()


def mutate(name, angle=.05):
    b[name].matrix_basis = b[name].matrix_basis @ Matrix.Rotation(angle,4,"X")
    bpy.context.view_layer.update()


def exact():
    return max(float(np.max(abs(snapshot(arm)[n]-v))) for n,v in initial.items()) < 1e-6


def test_protected_leg_rejected_during_arms():
    def bad(a,m): mutate("LeftLeg"); return {"accepted":True}
    r = run_refine_v4(arm,meshes,arm_stage=Stage("arms",("left",),bad),final_check=lambda *args:{"accepted":True})
    assert not r["accepted"] and "protected_joint_changed" in r["reasons"][0] and exact()
    return r


def test_second_stage_failure_restores_leg_and_fingers():
    def leg(a,m): mutate("LeftLeg"); return {"accepted":True}
    def arms(a,m): mutate("LeftHandIndex1"); return {"accepted":False,"reasons":["contact_failed"]}
    r = run_refine_v4(arm,meshes,leg_stage=Stage("legs",("left",),leg),
                     arm_stage=Stage("arms",("left",),arms), final_check=lambda *args:{"accepted":True})
    assert not r["accepted"] and exact()
    return r


def test_current_knee_read_after_leg_stage_and_final_rejection():
    observed = []
    def leg(a,m): mutate("LeftUpLeg"); return {"accepted":True}
    def arms(a,m):
        observed.append(np.array(arm.matrix_world@b["LeftLeg"].head))
        mutate("LeftArm"); return {"accepted":True}
    r = run_refine_v4(arm,meshes,leg_stage=Stage("legs",("left",),leg),
                     arm_stage=Stage("arms",("left",),arms), final_check=lambda *args:{"accepted":False,"reasons":["contact_drift"]})
    assert not np.allclose(observed[0],initial[b["LeftLeg"].name][:3,3])
    assert not r["accepted"] and exact()
    return r


def test_native_leg_recipe_recovers_perturbed_knee_with_real_skin():
    import hashlib
    plan = plan_legs("한쪽 무릎을 굽혀 앉기",selection="left",source="user_confirmed",reference="regression_fixture")
    part = extract_native_leg_part(arm,plan,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    mutate("LeftLeg", np.radians(5))
    before = inspect_self(arm)
    stage = Stage("legs",("left",),lambda a,m:apply_native_leg_part(a,m,part,plan,skin_check=inspect_self))
    r = run_refine_v4(arm,meshes,leg_stage=stage,
                     final_check=lambda a,m,receipts:skin_nonregression(before,inspect_self(a)))
    assert r["accepted"] and r["status"] == "applied", r
    # Opposite leg and upper body are protected by the transaction guard.
    return r


def test_missing_final_check_does_not_run_callback():
    def forbidden(a,m): raise AssertionError("must not run")
    r = run_refine_v4(arm,meshes,leg_stage=Stage("legs",("right",),forbidden))
    assert not r["accepted"] and exact()
    return r


def test_actual_arm_adapter_after_actual_leg_recovery():
    import hashlib
    from src.arm_refine.v4_adapter import apply_arm_stage,check_arm_stage
    profile=json.loads(profile_file.read_text())
    plan=plan_legs("한쪽 무릎을 굽혀 앉기",selection="left",
                   source="user_confirmed",reference="artificial_5_degree_regression_fixture")
    part=extract_native_leg_part(arm,plan,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    mutate("LeftLeg",np.radians(5));skin_before=inspect_self(arm)
    legs=Stage("legs",("left",),lambda a,m:apply_native_leg_part(a,m,part,plan,skin_check=inspect_self))
    arms=Stage("arms",("left","right"),lambda a,m:apply_arm_stage(a,m,
        clauses={"left":"세운 무릎 위에 손을 올린다","right":"반대 손을 아래로 내린다"},
        skin_check=inspect_self,rig_profile=profile,skeleton_scores=[0]*17))
    def final(a,m,stages):
        skin=skin_nonregression(skin_before,inspect_self(a))
        contact=check_arm_stage(a,m,stages[1]["receipt"],skin_check=inspect_self,rig_profile=profile)
        return {"accepted":skin["accepted"] and contact["accepted"],"skin":skin,"contact":contact}
    r=run_refine_v4(arm,meshes,leg_stage=legs,arm_stage=arms,final_check=final)
    assert r["accepted"] and all(s["changed_joints"] for s in r["stages"]),r
    return r


out.mkdir(parents=True,exist_ok=False)
for name, fn in list(globals().items()):
    if name.startswith("test_"):
        reset(); result = fn(); rows.append({"test":name,"passed":True,"receipt":result})
        print("PASS",name,flush=True)
(out/"native-tests.json").write_text(json.dumps({"tests":rows,"artificial_perturbations":True,
    "benchmark_score_evidence":False},ensure_ascii=False,indent=2)+"\n")
