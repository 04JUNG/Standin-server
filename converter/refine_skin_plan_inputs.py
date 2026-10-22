"""Blender-only planning inputs from a real failed finite FBX QA run.

The hand preview is explicitly unapproved. It only supplies forward rig frames;
the final QA recreates the derived asset and independently verifies every vertex.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    from converter.refine_v3_safe_snapshot_qa import snapshot, same_rig, sha
    from converter.hands import bake_hands
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--character", type=Path, required=True)
    p.add_argument("--hand-left", required=True)
    p.add_argument("--hand-right", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index("--")+1:])
    a.output_dir.mkdir(parents=True, exist_ok=False)
    rt.import_character = import_full_character
    native = list((a.qa_dir/"native-neutral").glob("*/model.fbx"))
    if len(native) != 1:
        raise ValueError("ambiguous_native_neutral")
    neutral = snapshot(native[0])
    rest = snapshot(a.qa_dir/"derived-neutral.fbx")
    body = snapshot(a.qa_dir/"body.fbx")
    same_rig(neutral, rest, require_rest=True)
    same_rig(rest, body)
    hand_path = a.output_dir/"unapproved-hand-planning.fbx"
    shutil.copyfile(a.qa_dir/"body.fbx", hand_path)
    report = bake_hands(hand_path, {"left": a.hand_left, "right": a.hand_right})
    hand = snapshot(hand_path)
    same_rig(body, hand)
    np.savez(a.output_dir/"inputs.npz", neutral=neutral["vertices"], rest=rest["vertices"],
        faces=rest["faces"], weights=rest["weights"], bones=rest["bones"],
        body_bones=body["bones"], hand_bones=hand["bones"], names=np.array(rest["names"]),
        parents=json.dumps(rest["parents"], sort_keys=True),
        source_sha256=sha(a.qa_dir/"source.bvh"), asset_sha256=sha(a.character),
        hand_left=a.hand_left, hand_right=a.hand_right)
    (a.output_dir/"planning.json").write_text(json.dumps(dict(
        source_sha256=sha(a.qa_dir/"source.bvh"), body_sha256=sha(a.qa_dir/"body.fbx"),
        unapproved_hand_sha256=sha(hand_path), hands=report, final_receipt_issued=False), indent=2))


if __name__ == "__main__":
    main()
