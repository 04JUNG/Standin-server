"""Smallest clear generated-phone correction; not library refine or pose assembly."""
import hashlib
from time import monotonic


class GeneratedHandSpacingRepair:
    def __init__(self, solver, *, clock=monotonic):
        self.solver, self.clock = solver, clock

    def filter(self, candidates, generation, required, skin_session):
        from .pose_requirements import audit_generated
        accepted, trials = [], []
        # One deadline and four trials for the whole request, not per candidate.
        deadline, count = self.clock() + 30., 0
        for candidate in candidates:
            original = skin_session.filter([candidate], generated=True, limit=1)
            if original:
                accepted.extend(original)
                continue
            text = candidate.get('bvh')
            if not isinstance(text, str) or not text:
                continue
            original_screen = skin_session.check(text)  # cached, no second conversion
            if (original_screen['status'] != 'intersection' or not required
                    or not required.get('active') or required.get('hand_counts', {}).get('phone') != 'two'):
                continue
            source_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if source_hash != candidate.get('bvh_sha256'):
                continue
            for delta in (2., 4., 6., 8.):
                remaining = deadline - self.clock()
                if remaining <= 0 or count >= 4 or skin_session.failed:
                    break
                count += 1
                note = dict(source_bvh_sha256=source_hash, delta_cm=delta, status='rejected')
                trials.append(note)
                try:
                    result = self.solver(candidate, generation, delta, remaining)
                    if result.get('passed') is not True:
                        note['reason'] = 'ik_source_or_safety_failed'
                        continue
                    repaired = result['candidate']
                    validation = repaired['validation']
                    if (validation.get('non_arm_channels_preserved') is not True
                            or not all(validation.get(k) is True for k in ('ik_and_geometry', 'saved_file', 'independent'))):
                        raise ValueError('incomplete_repair_validation')
                    audit = audit_generated(required, repaired)
                    note['conditions'] = audit
                    if audit['status'] != 'passed':
                        note['reason'] = 'observed_conditions_failed'
                        continue
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        note['reason'] = 'repair_deadline_exhausted'
                        break
                    receipt = skin_session.check(repaired['bvh'], timeout_seconds=remaining)
                    note['hand_skin_screen'] = receipt
                    if self.clock() >= deadline:
                        note['reason'] = 'repair_deadline_exhausted'
                        break
                    if receipt['status'] != 'clear_screen':
                        note['reason'] = 'repaired_hand_skin_not_clear'
                        if receipt['status'] == 'unavailable':
                            # Feed the unknown through the normal gate so the
                            # outer search returns error, never a false empty.
                            skin_session.filter([repaired], generated=True, limit=1)
                            break
                        continue
                    validation['whole_pose_conditions'] = audit
                    repaired.update(kind='generated_preview', hand_skin_screen=receipt,
                        hand_spacing_repair=dict(source_bvh_sha256=source_hash, delta_cm=delta,
                            smallest_clear_tested_delta=True, semantic_grip_certified=False),
                        refine_allowed=False, release_allowed=False, lookup_allowed=False,
                        actual_mesh_validated=False, review_required=True)
                    accepted.append(repaired)
                    note['status'] = 'accepted_review_only'
                    break
                except Exception:
                    note['reason'] = 'repair_validation_unavailable'
        return accepted, dict(version='bounded-generated-phone-spacing-v1',
            budget_seconds=30., max_trials=4, trials=trials, model_calls=0,
            library_poses_modified=False, complete_collision_certificate=False)
