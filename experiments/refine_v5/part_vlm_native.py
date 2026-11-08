"""Save the VLM-goal pose after an FBX readback and native skin audit."""

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.refine_v5.ab_blender import apply_rotations, bone_world, sha, verify_plan, weighted_mesh
from experiments.refine_v5.ab_native_audit import (
    RoundtripError, export_checked, frozen_contacts, measure_contact, snapshot, strict_skin, triage, write,
)
from experiments.refine_v5.audit_iterative_native import load
from experiments.refine_v5.target_safety import skin_nonregression


CAPS = {side + joint: limit for side in ('Left', 'Right') for joint, limit in
        [('Arm', 90.), ('ForeArm', 105.), ('UpLeg', 105.), ('Leg', 115.), ('Foot', 78.)]}


def read(path):
    return json.loads(path.read_text())


def contact_notes(goals):
    return [(limb.split('_')[0].title(), 'waist' if goal['contact'] == 'clear_of_waist' else goal['contact'])
            for limb, goal in goals.items() if goal['contact'] != 'unknown']


def validate(folder, case):
    bound = read(folder / 'vlm-plan.json')
    if (read(folder / 'input.json') != case or sha(folder / 'input.json') != bound['input_sha256'] or
            sha(folder / 'vlm-response.json') != bound['response_sha256'] or
            sha(case['source']) != case['source_sha256']):
        raise ValueError('bound input changed')
    proposal = bound['result']
    if proposal and proposal['status'] == 'proposal':
        if proposal['source_sha256'] != case['source_sha256']:
            raise ValueError('proposal belongs to another B0')
        for name, value in proposal['edits_deg'].items():
            vec = np.asarray(value)
            if name not in CAPS or vec.shape != (3,) or not np.isfinite(vec).all() or np.linalg.norm(vec) > CAPS[name] + 1e-5:
                raise ValueError('joint rotation outside VLM solver profile')
    return bound


def check_candidate(folder, case, bound):
    saved = folder / 'vlm.fbx'
    source = Path(case['source'])
    proposal = bound['result']
    arm, meshes = load(source)
    base = snapshot(arm)
    names = list(bone_world(arm))
    verts, faces, weights = weighted_mesh(meshes, names)
    notes = contact_notes(bound['goals'])
    world = np.array([base['matrices'][name] for name in names])
    patches = frozen_contacts(verts, faces, weights, names, world, notes)
    contact_before = measure_contact(verts, faces, patches)
    contact_candidate = []
    accepted = False
    reasons = []
    drift, cloud, frame_error = None, None, None
    bones, final_v, final_f = base['matrices'], verts, faces
    skin_base, skin_candidate, proposed_bones = None, None, None
    if not proposal or proposal['status'] != 'proposal':
        reasons.append(proposal['status'] if proposal else 'no_supported_goal')
    else:
        skin_base = triage(arm)
        affected = set(proposal['edits_deg'])
        for bone in case['bones']:
            if base['parents'][bone['name']] in affected:
                affected.add(bone['name'])
        protected = set(names) - affected
        apply_rotations(arm, proposal['edits_deg'])
        verify_plan(arm, proposal['expected_heads'])
        with TemporaryDirectory(dir=folder) as temp:
            path = Path(temp) / 'candidate.fbx'
            try:
                arm, v, f, _, cloud, frame_error = export_checked(path, arm, meshes, names, base, load)
                actual = bone_world(arm)
                drift = float(max(np.max(np.abs(actual[n] - base['matrices'][n])) for n in protected))
                if drift > 2e-5:
                    raise RoundtripError('protected pose changed')
                if not np.array_equal(f, faces):
                    raise RoundtripError('saved surface indexing changed')
                np.savez_compressed(folder / 'vlm-proposal.npz', vertices=v, faces=f)
                proposed_bones = [dict(name=name, matrix=actual[name].tolist()) for name in names]
                contact_candidate = measure_contact(v, f, patches)
                skin_candidate = triage(arm)
                if 'unavailable' in skin_base or 'unavailable' in skin_candidate:
                    reasons.append('skin_triage_unavailable')
                else:
                    reasons.extend(skin_nonregression(skin_base, skin_candidate))
                if not reasons:
                    path.replace(saved)
                    bones, final_v, final_f = actual, v, f
                    accepted = True
            except RoundtripError as exc:
                reasons.append(str(exc))
    if not accepted:
        shutil.copyfile(source, saved)
    np.savez_compressed(folder / 'vlm.npz', vertices=final_v, faces=final_f)
    skin = strict_skin(final_v, final_f, case['body_asset_sha256'], full_body=True)
    row = dict(accepted=accepted, reasons=reasons, source_sha256=sha(source), sha256=sha(saved),
               bones=[dict(name=name, matrix=np.asarray(bones[name]).tolist()) for name in names],
               proposed_bones=proposed_bones,
               protected_matrix_drift=drift, cloud_m=cloud, frame_error=frame_error,
               contact_before=contact_before, contact_candidate=contact_candidate,
               contact_saved=contact_candidate if accepted else contact_before,
               baseline_regional_collisions=skin_base.get('collisions', []) if skin_base else None,
               candidate_regional_collisions=skin_candidate.get('collisions', []) if skin_candidate else None,
               full_skin_status=skin['status'], full_skin_certified=skin['complete_collision_certificate'])
    write(folder / 'vlm-audit.json', row)
    print('VLM_FBX', case['case_id'], accepted, reasons, skin['status'], flush=True)


def measure_reference(path, expected, notes):
    if sha(path) != expected:
        raise ValueError('reference FBX bytes changed')
    if not notes:
        return []
    arm, meshes = load(path)
    world = bone_world(arm)
    names = list(world)
    verts, faces, weights = weighted_mesh(meshes, names)
    return measure_contact(verts, faces, frozen_contacts(verts, faces, weights, names, np.array(list(world.values())), notes))


def run(root, control, gt_dir):
    cases = read(root / 'manifest.json')['cases']
    for case in cases:
        folder = root / case['case_id']
        existing = folder / 'vlm-audit.json'
        if not existing.exists() or 'proposed_bones' not in read(existing):
            check_candidate(folder, case, validate(folder, case))
    # Only after every candidate and its acceptance are frozen.
    for case in cases:
        cid = case['case_id']
        folder = root / cid
        previous = control / cid
        goals = read(folder / 'vlm-plan.json')['goals']
        notes = contact_notes(goals)
        baseline = read(previous / 'detector/audit.json')
        gt = read(previous / 'gt.json')
        detector = previous / 'detector' / f'{cid}-part-goals-head-limbs.fbx'
        write(folder / 'reference-contact.json', {
            'detector': measure_reference(detector, baseline['output_sha256'], notes),
            'ground_truth': measure_reference(gt_dir / f'{cid}.fbx', gt['sha256'], notes),
        })


if __name__ == '__main__':
    run(*map(Path, sys.argv[sys.argv.index('--') + 1:]))
