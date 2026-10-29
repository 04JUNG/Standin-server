"""Real Blender regression; uses local library assets, never downloads/publishes.

blender -b --python-exit-code 1 --python tests/converter/verify_source_fingers.py \
  -- --library /path/to/library --character /path/to/master.fbx --out /tmp/hand-qa
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy
from converter import retarget as rt
from converter.convert import convert
from converter.bone_map import PROFILES
from converter.source_fingers import bake_source_fingers, bone_names
from converter.hands import bake_hands
from converter.worker import run_job


def snapshot(path):
    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    body = {b.name: tuple(v for row in b.matrix_local for v in row)
            for b in arm.data.bones if b.name in PROFILES["mixamo"].values()}
    tips = {}
    for side in ("left", "right"):
        wrist, fingers = bone_names(arm.data.bones.keys(), side)
        pivot = arm.matrix_world @ arm.data.bones[wrist].head_local
        tips[side] = {(f + str(j)): ((arm.matrix_world @ arm.data.bones[n].tail_local) - pivot).length
                      for (f, j), n in fingers.items()}
    return body, tips


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    ratios = {}
    cases = [("Boxing_01", False, None), ("a3_cough_both_hands", False, None),
             ("a3_cough_both_hands", True, None), ("a1_open_arms_explain", False, None),
             ("a3_cough_both_hands", False, {"left": "open", "right": None}),
             ("cmu_91_51_00092_mirror", False, None),
             ("UAL2__Zombie_Walk_Fwd_Loop__p01_f0002_mirror", False, None),
             ("style100_WildLegs_FW_f004698_eed5ec3a", False, None)]
    for index, (pose, mirror, hands) in enumerate(cases):
        source = args.library / "bvh" / (pose + ".bvh")
        folder = args.out / str(index)
        folder.mkdir(exist_ok=True)
        before_path, after_path = folder / "before.fbx", folder / "after.fbx"
        body = convert(bvh_path=str(source), character_fbx=str(args.character),
                       out_path=str(before_path), embed_textures=False, mirror=mirror)
        assert body.ok, body.as_dict()
        before_body, before = snapshot(before_path)
        shutil.copyfile(before_path, after_path)
        receipt = bake_source_fingers(after_path, source, mirror=mirror, hands=hands)
        if hands:
            assert receipt["sides"]["left"]["status"] == "preset_override"
            assert receipt["applied_bones"] == 15
            _, intermediate = snapshot(after_path)
            assert abs(intermediate["left"]["index3"] - before["left"]["index3"]) < 1e-5
            bake_hands(after_path, hands)
        after_body, after = snapshot(after_path)
        assert before_body.keys() == after_body.keys()
        error = max(abs(a-b) for n in before_body for a, b in zip(before_body[n], after_body[n]))
        # FBX rebakes normalize quaternions again; allow 0.1 mm / 1e-4
        # matrix-element roundtrip error. The in-memory stage checks 1e-5.
        assert error < 1e-4, (pose, "body changed", error)
        ratio = {side: after[side]["index3"] / before[side]["index3"] for side in before}
        if pose == "Boxing_01":
            assert max(ratio.values()) < .65, ("boxing fingers did not curl", ratio)
        if pose == "a1_open_arms_explain":
            assert min(ratio.values()) > .9, ("open fingers collapsed", ratio)
        if pose.startswith("cmu_"):
            assert receipt["status"] == "source_unavailable"
            assert before_path.read_bytes() == after_path.read_bytes()
        else:
            assert receipt["applied_bones"] == (15 if hands else 30)
        if pose == "a3_cough_both_hands" and not hands:
            ratios[mirror] = ratio
        results.append(dict(pose=pose, mirror=mirror, hands=hands, receipt=receipt,
                            body_matrix_max_error=error, index_tip_length_ratio=ratio))
    assert abs(ratios[False]["left"] - ratios[True]["right"]) < .08
    assert abs(ratios[False]["right"] - ratios[True]["left"]) < .08

    # Exercise the actual HTTP worker path, including crop and rendering the
    # reimported final FBX. This catches integration bypasses beyond the helper.
    folder = args.out / "live-worker"
    folder.mkdir(exist_ok=True)
    output = folder / "artifact.fbx"
    job = dict(bvh_path=str(args.library / "bvh/Boxing_01.bvh"),
        character_fbx=str(args.character), character_sha256=hashlib.sha256(args.character.read_bytes()).hexdigest(),
        output_path=str(output), temp_dir=str(folder), mirror=False, force_exact_v324=False,
        shape_recipe=None, hands=None, output_scope="half", preview_view="front",
        camera_rotation=None, conversion_id="source-hand-regression", character_id="standin-master-v2",
        thumbnail=None)
    report = run_job(job)
    assert report["ok"] and report["source_fingers"]["applied_bones"] == 30
    assert report["artifact_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["preview_sha256"] == hashlib.sha256((folder / "preview.png").read_bytes()).hexdigest()
    assert (folder / "preview.png").read_bytes().startswith(b"\x89PNG")
    (args.out / "verification.json").write_text(json.dumps(
        dict(passed=True, cases=results, live_worker=report), indent=2))
    print("SOURCE_FINGERS_VERIFIED", len(results), "cases plus live worker", flush=True)


if __name__ == "__main__":
    main()
