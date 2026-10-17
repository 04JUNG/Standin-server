"""Offline bounds, failure policy and unchanged non-arm channels."""
from copy import deepcopy
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from src.hand_spacing_repair import GeneratedHandSpacingRepair
from tests.test_hand_skin_gate import gate


REQUIRED = dict(active=True, hand_counts={'phone': 'two'})


def candidate(text='source'):
    return dict(bvh=text, bvh_sha256=hashlib.sha256(text.encode()).hexdigest(),
        validation=dict(ik_and_geometry=True, saved_file=True, independent=True,
                        non_arm_channels_preserved=True))


class Solver:
    def __init__(self): self.calls = []
    def __call__(self, original, generation, delta, remaining):
        self.calls.append((delta, remaining))
        return dict(passed=True, candidate=candidate('repaired-'+str(delta)))


@pytest.fixture(autouse=True)
def audit(monkeypatch):
    import src.pose_requirements as module
    monkeypatch.setattr(module, 'audit_generated', lambda *a: {'status': 'passed'})


def run(tmp_path, statuses, *, required=REQUIRED, solver=None, clock=None, candidates=None):
    solver = solver or Solver()
    repair = GeneratedHandSpacingRepair(solver, **({'clock': clock} if clock else {}))
    session = gate(tmp_path, statuses).session({'left': 'relaxed', 'right': 'relaxed'})
    rows, report = repair.filter(candidates or [candidate()], {}, required, session)
    return rows, report, solver, session


@pytest.mark.parametrize('status', ['clear_screen', 'unavailable'])
def test_only_known_intersections_trigger_repair(tmp_path, status):
    rows, report, solver, session = run(tmp_path, [status])
    assert bool(rows) == (status == 'clear_screen')
    assert not solver.calls and not report['trials']
    if status == 'unavailable': assert session.unverified


@pytest.mark.parametrize('required', [None, {}, dict(active=False, hand_counts={'phone': 'two'}),
    dict(active=True, hand_counts={'phone': 'one'}), dict(active=True, hand_counts={})])
def test_no_guessing_bilateral_phone_condition(tmp_path, required):
    rows, _, solver, _ = run(tmp_path, ['intersection'], required=required)
    assert not rows and not solver.calls


def test_smallest_clear_delta_and_same_hands_no_permission_upgrade(tmp_path):
    original = candidate(); before = deepcopy(original)
    rows, report, solver, session = run(tmp_path,
        ['intersection', 'intersection', 'clear_screen'], candidates=[original])
    assert [d for d, _ in solver.calls] == [2., 4.]
    assert all(0 < t <= 30 for _, t in solver.calls)
    [pick] = rows
    assert pick['hand_spacing_repair']['delta_cm'] == 4.
    assert pick['hand_spacing_repair']['source_bvh_sha256'] == original['bvh_sha256']
    assert pick['bvh_sha256'] != original['bvh_sha256']
    assert not pick['refine_allowed'] and not pick['release_allowed']
    assert not pick['lookup_allowed'] and not pick['actual_mesh_validated']
    assert report['model_calls'] == 0 and original == before
    assert all(call[1] == session.hands for call in session.gate.client.calls)


def test_four_trials_shared_across_all_candidates(tmp_path):
    rows, report, solver, _ = run(tmp_path, ['intersection']*10,
                                 candidates=[candidate(), candidate('second')])
    assert not rows and len(report['trials']) == 4
    assert [d for d, _ in solver.calls] == [2., 4., 6., 8.]


def test_unknown_repair_screen_stops_and_remains_error_evidence(tmp_path):
    rows, _, solver, session = run(tmp_path, ['intersection', 'unavailable'])
    assert not rows and len(solver.calls) == 1 and session.unverified


def test_condition_failure_never_calls_mesh_checker(tmp_path, monkeypatch):
    import src.pose_requirements as module
    monkeypatch.setattr(module, 'audit_generated', lambda *a: {'status': 'near_match'})
    rows, report, solver, session = run(tmp_path, ['intersection'])
    assert not rows and len(solver.calls) == 4
    assert len(session.gate.client.calls) == 1
    assert all(t['reason'] == 'observed_conditions_failed' for t in report['trials'])


@pytest.mark.parametrize('bad', ['ik_and_geometry', 'saved_file', 'independent', 'non_arm_channels_preserved'])
def test_incomplete_validation_never_reaches_mesh(tmp_path, bad):
    def solver(*args):
        c = candidate('new'); c['validation'][bad] = False
        return dict(passed=True, candidate=c)
    rows, _, _, session = run(tmp_path, ['intersection'], solver=solver)
    assert not rows and len(session.gate.client.calls) == 1


def test_deadline_blocks_late_success(tmp_path):
    now = [0.]
    def solver(*args):
        now[0] = 31.
        return dict(passed=True, candidate=candidate('new'))
    rows, report, _, session = run(tmp_path, ['intersection'], solver=solver, clock=lambda: now[0])
    assert not rows and len(report['trials']) == 1 and len(session.gate.client.calls) == 1


def test_subprocess_is_offline_and_killable(tmp_path, monkeypatch):
    import src.natural_pose as module
    monkeypatch.setattr(module, 'file_hash', lambda *a: module.RIG_SHA256)
    c = candidate(); c['search_v2'] = {'description': {'text': '양손 휴대폰'}}
    g = {'diagnostics': {'attempts': [{'status': 'passed', 'proposal': {}}]}}
    calls = []
    def timeout(argv, **kwargs):
        calls.append(argv)
        assert argv[-1] == 'src.hand_spacing_worker'
        assert kwargs['timeout'] == 3
        assert json.loads(kwargs['input'])['source_bvh_sha256'] == c['bvh_sha256']
        raise subprocess.TimeoutExpired('repair', 3)
    monkeypatch.setattr(subprocess, 'run', timeout)
    out = module.NaturalLanguagePoseGenerator(tmp_path/'rig', tmp_path/'taxonomy').repair_hand_spacing(c, g, 2., 3.)
    assert out['passed'] is False and out['reasons'] == ['hand_repair_deadline_exhausted']
    assert len(calls) == 1


def tiny_bvh(root_value=0, arm_value=0):
    return ('HIERARCHY\nROOT Hips { OFFSET 0 0 0 CHANNELS 1 Xrotation '
            'JOINT LeftArm { OFFSET 1 0 0 CHANNELS 1 Xrotation End Site { OFFSET 1 0 0 } } } '
            f'MOTION\nFrames: 1\nFrame Time: 0.033\n{root_value} {arm_value}\n')


def test_non_arm_change_rejected():
    from src.hand_spacing_worker import arm_only_change
    arm_only_change(tiny_bvh(), tiny_bvh(arm_value=2))
    with pytest.raises(ValueError, match='non_arm'):
        arm_only_change(tiny_bvh(), tiny_bvh(root_value=2))


def test_normal_library_selection_is_not_repaired(tmp_path):
    from tests.test_fullbody_routing import engine, decision, evidence
    e = engine(); e.hand_skin_gate = gate(tmp_path)
    def forbidden(*args): pytest.fail('library poses must not be edited')
    e.hand_spacing_repairer = GeneratedHandSpacingRepair(forbidden)
    result = e.search(decision(), evidence())
    assert result['selected'] and e.generator.calls == 0


@pytest.mark.parametrize('mesh_status', ['clear_screen', 'unavailable'])
def test_fullbody_generation_uses_repair_without_promoting_export(tmp_path, mesh_status):
    from tests.test_integrated_recovery import constrained_scene
    from src.experimental.text_pose_generation import GenerationResult
    from api.models import DynamicSearchOut
    e, d, ev, _ = constrained_scene()
    e.generator = SimpleNamespace(generate=lambda description: GenerationResult(
        'preview_ready', candidates=[candidate()], diagnostics={}))
    e.hand_skin_gate = gate(tmp_path, ['intersection', mesh_status], ids=('a',))
    e.hand_spacing_repairer = GeneratedHandSpacingRepair(Solver())
    result = DynamicSearchOut.model_validate(e.search(d, ev)).model_dump()
    assert not result['selected']
    if mesh_status == 'clear_screen':
        assert result['status'] == 'no_result'
        g = result['generation']
        assert g['status'] == 'preview_ready' and not g['release_allowed']
        assert g['candidates'][0]['hand_spacing_repair']['delta_cm'] == 2.
        assert g['diagnostics']['hand_spacing_repair']['model_calls'] == 0
    else:
        assert result['status'] == 'error'
        assert result['generation']['status'] == 'rejected'
        assert not result['generation']['candidates']
