"""Run the existing independent FBX inspector at its verified native pose.

The source's shoulder projection has already run. Preserve its signature during
surface-only replay, as reviewed in CSP PR #155, and reject any bone drift.
No solver, source surface, correlation or error threshold is changed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import inspect_shape_bake as inspector

original_loader = inspector.controller


def controller(recipe):
    module = original_loader(recipe)
    original_correct = module.correct_pose

    def correct(*args, **kwargs):
        _, rig = module.objects()
        before = {b.name: b.matrix.copy() for b in rig.pose.bones}
        if callable(getattr(module, 'sl_signature', None)):
            rig['standin_shoulder_signature'] = module.sl_signature(rig)
        result = original_correct(*args, **kwargs)
        for b in rig.pose.bones:
            error = max(abs(x - y) for a, c in zip(b.matrix, before[b.name]) for x, y in zip(a, c))
            if error > 1e-7:
                raise ValueError('surface replay changed verified native pose')
        return result

    module.correct_pose = correct
    return module


inspector.controller = controller
if __name__ == '__main__':
    raise SystemExit(inspector.main())
