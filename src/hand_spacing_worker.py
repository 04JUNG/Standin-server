"""Killable, offline bilateral-phone IK repair. Never calls a model or edits a library."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from .natural_pose import RIG_SHA256, file_hash


def arm_only_change(before, after):
    import numpy as np
    from .bvh import parse_bvh, channel_starts
    allowed = {side + part for side in ('Left', 'Right')
               for part in ('Shoulder', 'Arm', 'ForeArm', 'Hand')}
    with tempfile.TemporaryDirectory(prefix='hand-repair-audit-') as folder:
        paths = [Path(folder)/name for name in ('before.bvh', 'after.bvh')]
        for path, text in zip(paths, (before, after)):
            path.write_text(text, encoding='utf-8')
        (joints, frames), (other, repaired) = [parse_bvh(str(p)) for p in paths]
    if len(joints) != len(other) or frames.shape != repaired.shape or len(frames) != 1:
        raise ValueError('repair_hierarchy_changed')
    starts = channel_starts(joints)
    for i, (a, b) in enumerate(zip(joints, other)):
        if (a[0] != b[0] or a[1] != b[1] or a[3:] != b[3:]
                or not np.array_equal(a[2], b[2])):
            raise ValueError('repair_hierarchy_changed')
        if a[0] not in allowed:
            sl = slice(starts[i], starts[i] + len(a[3]))
            if not np.allclose(frames[:, sl], repaired[:, sl], atol=1e-6, rtol=0):
                raise ValueError('repair_changed_non_arm_channels')


def solve_repair(job):
    from .natural_pose_schema import validate_proposal
    from .natural_pose_worker import solve_proposal
    delta = job['delta_cm']
    if type(delta) not in (float, int) or delta not in (2., 4., 6., 8.):
        raise ValueError('repair_delta_out_of_bounds')
    if file_hash(os.environ['POSE_GENERATION_RIG']) != RIG_SHA256:
        raise ValueError('repair_rig_mismatch')
    taxonomy_path = Path(job['taxonomy'])
    if file_hash(taxonomy_path) != job['taxonomy_sha256']:
        raise ValueError('repair_taxonomy_mismatch')
    taxonomy = json.loads(taxonomy_path.read_text())
    proposal = validate_proposal(job['proposal'], taxonomy)
    text = job['text']
    if not isinstance(text, str) or not text or len(text) > 4000:
        raise ValueError('repair_invalid_text')
    # Reproduce the *exact* source before considering a correction. A stale
    # proposal, previous leg correction, or changed solver cannot repair a
    # different BVH under the old identity.
    source = solve_proposal(proposal, text)
    if (not source['passed'] or source['candidate']['bvh_sha256'] != job['source_bvh_sha256']):
        raise ValueError('repair_source_proposal_mismatch')
    result = solve_proposal(proposal, text, arm_spacing_delta=delta)
    if result['passed']:
        candidate = result['candidate']
        arm_only_change(source['candidate']['bvh'], candidate['bvh'])
        if hashlib.sha256(candidate['bvh'].encode()).hexdigest() != candidate['bvh_sha256']:
            raise ValueError('repair_output_hash_mismatch')
        candidate['validation']['non_arm_channels_preserved'] = True
    if (file_hash(os.environ['POSE_GENERATION_RIG']) != RIG_SHA256
            or file_hash(taxonomy_path) != job['taxonomy_sha256']):
        raise ValueError('repair_source_changed')
    return result


def main():
    try:
        raw = sys.stdin.read(256_001)
        if len(raw) > 256_000:
            raise ValueError('repair_job_too_large')
        result = solve_repair(json.loads(raw))
    except Exception as error:
        result = dict(passed=False, reasons=['hand_repair_worker:' + type(error).__name__])
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
