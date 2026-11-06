"""Blender-only independent readback; existing candidate bytes are never changed."""
import bpy, json, sys
from pathlib import Path

def main():
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from experiments.refine_v5.native_contract_v2 import compare_native
    root = Path(sys.argv[sys.argv.index('--') + 1])
    goals = json.loads((root / 'review-v2/goals-common.json').read_text())['records']
    verifications = json.loads((root / 'review-v2/native-goal-preserved/verification.json').read_text())
    results = []

    def read(path):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(path), ignore_leaf_bones=False)
        arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE'))
        heads = {}
        parents = {}
        lengths = {}
        for b in arm.pose.bones:
            n = b.name.split(':')[-1]
            heads[n] = list(arm.matrix_world @ b.head)
            parents[n] = b.parent.name.split(':')[-1] if b.parent else None
            lengths[n] = (arm.matrix_world @ (b.tail - b.head)).length
        return (heads, parents, lengths)
    for row in verifications:
        goal = next((g for g in goals if g['case_id'] == row['case_id'] and g['mode'] == 'human_fbx_2d_projection' and (g['profile'] == 'upper60')))
        _, parents, lengths = read(goal['base_fbx'])
        actual, ap, al = read(row['artifact'])
        audit = compare_native(goal['planned_heads'], actual, parents, ap, lengths, al)
        results.append(dict(case_id=row['case_id'], preserved_bones=len(actual), **audit, mesh_checked=False))
    (root / 'review-v2/native-goal-preserved/independent-readback.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results))
if __name__ == '__main__':
    main()
