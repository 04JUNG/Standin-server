"""Export and reload the independent head candidates on native assets."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


def audit(root, out):
    from experiments.refine_v5.ab_blender import apply_rotations, bone_world, sha, verify_plan
    from experiments.refine_v5.ab_native_audit import RoundtripError, export_checked, snapshot, triage, write
    from experiments.refine_v5.audit_iterative_native import load
    from experiments.refine_v5.part_head import HEAD_CAP_DEG, NECK_CAP_DEG, orientation_errors
    from experiments.refine_v5.target_safety import skin_nonregression

    manifest = root / 'manifest.json'
    frozen = json.loads((root / 'head-proposals.json').read_text())
    if sha(manifest) != frozen['manifest_sha256']:
        raise ValueError('head inputs differ from frozen proposal')
    cases = {c['case_id']: c for c in json.loads(manifest.read_text())['cases']}
    out.mkdir(exist_ok=False, parents=True)
    rows = []
    for proposal in frozen['results']:
        case = cases[proposal['case_id']]
        cid, source = case['case_id'], case['source']
        if sha(source) != case['source_sha256'] or sha(case['image']) != case['image_sha256']:
            raise ValueError('head input changed')
        edits = proposal['edits_deg']
        if not edits or set(edits) - {'Head', 'Neck'} or ('Neck' in edits and 'neck' not in case['part_goal']['head'].get('axes_body', {})):
            raise ValueError('unapproved head edits')
        caps = {'Head': HEAD_CAP_DEG, 'Neck': NECK_CAP_DEG}
        if any(np.asarray(edit).shape != (3,) or not np.isfinite(edit).all()
               or np.linalg.norm(edit) > caps[name] + 1e-5 for name, edit in edits.items()):
            raise ValueError('head edits exceed the profile')
        bones = case['bones']
        names = [b['name'] for b in bones]
        editable = set(edits)
        for bone in bones:
            if bone['parent'] in editable:
                editable.add(bone['name'])
        protected = set(names) - editable
        arm, meshes = load(source)
        baseline = snapshot(arm)
        base_skin = triage(arm)
        apply_rotations(arm, edits)
        verify_plan(arm, proposal['expected_heads'])
        world = bone_world(arm)
        if any(not np.allclose(world[name], baseline['matrices'][name], atol=1e-5, rtol=0) for name in protected):
            raise ValueError('head proposal changed unrelated bones')
        rejected = []
        with TemporaryDirectory(dir=out) as tmp:
            candidate = Path(tmp) / 'candidate.fbx'
            try:
                arm, _, _, _, cloud, frame = export_checked(candidate, arm, meshes, names, baseline, load)
            except RoundtripError as exc:
                rejected.append(str(exc))
            else:
                final = bone_world(arm)
                actual = orientation_errors(case, np.array([final[name] for name in names]), names)
                if any(abs(actual[key] - proposal['after_deg'][key]) > .05 for key in actual):
                    rejected.append('head_orientation_changed_after_save')
                region = triage(arm)
                if 'unavailable' in region or 'unavailable' in base_skin:
                    rejected.append('skin_triage_unavailable')
                else:
                    rejected.extend(skin_nonregression(base_skin, region))
                if not rejected:
                    saved = out / f'{cid}-head.diagnostic.fbx'
                    candidate.replace(saved)
        if rejected:
            saved, actual = Path(source), proposal['before_deg']
        rows.append(dict(case_id=cid, status='fallback_exact_B0' if rejected else 'diagnostic',
                         before_deg=proposal['before_deg'], after_deg=actual, reasons=rejected,
                         fbx=saved.name if not rejected else str(saved), sha256=sha(saved),
                         max_cloud_error_m=None if rejected else cloud,
                         max_frame_error=None if rejected else frame))
    write(out / 'head-audit.json', rows)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    root, out = sys.argv[sys.argv.index('--') + 1:]
    audit(Path(root), Path(out))
