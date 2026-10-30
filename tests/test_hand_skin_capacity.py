"""Capacity changes scheduling, never candidate order, receipts or permissions."""
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import threading
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from api.pose_presentation import build_pose_presentation
from src.config import Config
from src.fullbody_routing import FullBodySearch
from src.hand_skin_screen import HandSkinGate, SkinSession
from src.hand_skin_parallel import ParallelSkinSession
from tests.test_fullbody_routing import Camera, Semantic, decision, evidence
from tests.test_hand_skin_parallel import Checker, receipt
from tests.test_hand_skin_parallel_contracts import asset, normalize


def make_gate(tmp_path, checker, capacity, ids=('a', 'b', 'c', 'd', 'e', 'f')):
    import hashlib
    hashes = {}
    for pid in ids:
        data = 'BVH ' + pid
        (tmp_path / (pid + '.bvh')).write_text(data)
        hashes[pid] = hashlib.sha256(data.encode()).hexdigest()
    return HandSkinGate(checker, tmp_path, hashes, concurrency=capacity)


@pytest.mark.parametrize('capacity', range(1, 6))
def test_runtime_selects_requested_capacity(tmp_path, capacity):
    gate = make_gate(tmp_path, Checker(), capacity)
    assert type(gate.session()) is (SkinSession if capacity == 1 else ParallelSkinSession)
    if capacity > 1:
        assert gate.parallel_slots.capacity == capacity


@pytest.mark.parametrize('capacity', [0, 6, True, 2.5, '2'])
def test_config_rejects_invalid_capacity_at_startup(capacity):
    with pytest.raises(ValueError): Config(hand_skin_screen_concurrency=capacity)


@pytest.mark.parametrize('capacity', range(1, 6))
def test_app_installs_request_cancellation_for_every_parallel_capacity(capacity):
    script = '''
from api.app import app
from api.hand_skin_middleware import HandSkinMiddleware
from src.config import CFG
installed = any(item.cls is HandSkinMiddleware for item in app.user_middleware)
assert installed == (CFG.hand_skin_screen_concurrency > 1)
'''
    subprocess.run([sys.executable, '-c', script], check=True,
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        env={**os.environ, 'HAND_SKIN_SCREEN_CONCURRENCY': str(capacity)})


class BlockingChecker:
    def __init__(self):
        self.state = SimpleNamespace(active=0, peak=0, budgets=[],
            condition=threading.Condition(), release=threading.Event())

    def check(self, text, hands, timeout):
        state = self.state
        with state.condition:
            state.active += 1
            state.peak = max(state.peak, state.active)
            state.budgets.append(timeout)
            state.condition.notify_all()
        try:
            assert state.release.wait(2)
            return receipt(text, hands)
        finally:
            with state.condition:
                state.active -= 1
                state.condition.notify_all()


@pytest.mark.parametrize('capacity', range(2, 6))
def test_capacity_bounds_all_request_sessions(tmp_path, capacity):
    checker = BlockingChecker()
    gate = make_gate(tmp_path, checker, capacity)
    with ThreadPoolExecutor(capacity + 2) as pool:
        futures = [pool.submit(gate.session().filter, [{'pose_id': 'a'}])
                   for _ in range(capacity + 2)]
        try:
            with checker.state.condition:
                assert checker.state.condition.wait_for(
                    lambda: checker.state.active == capacity, timeout=1)
                assert checker.state.peak == capacity
        finally:
            checker.state.release.set()
        assert all(f.result(2)[0]['pose_id'] == 'a' for f in futures)
    assert checker.state.peak == capacity and gate.parallel_slots.active == 0


@pytest.mark.parametrize('capacity', range(2, 6))
def test_45_second_work_budget_never_shortens_calls_to_fill_capacity(tmp_path, capacity):
    checker = BlockingChecker()
    gate = make_gate(tmp_path, checker, capacity)
    session = gate.session()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(session.filter, [{'pose_id': p} for p in ('a', 'b', 'c', 'd', 'e')])
        try:
            with checker.state.condition:
                assert checker.state.condition.wait_for(
                    lambda: checker.state.active == min(capacity, 3), timeout=1)
                assert all(14.5 < value <= 15 for value in checker.state.budgets)
        finally:
            checker.state.release.set()
        assert [r['pose_id'] for r in future.result(2)] == ['a', 'b', 'c', 'd', 'e']
    assert checker.state.peak <= min(capacity, 3)


@pytest.mark.parametrize('capacity', range(2, 6))
@pytest.mark.parametrize('statuses', list(product(['clear_screen', 'intersection', 'unavailable'], repeat=3)))
def test_visible_results_match_serial_for_all_three_candidate_outcomes(tmp_path, capacity, statuses):
    results = []
    for workers in (1, capacity):
        checker = Checker(dict(zip(('a', 'b', 'c'), statuses)))
        gate = make_gate(tmp_path, checker, workers)
        engine = FullBodySearch([SimpleNamespace(pose_id=p, meta={}) for p in ('a', 'b', 'c')],
            Semantic(), Camera(), hand_skin_gate=gate)
        raw = engine.search(decision(), evidence())
        presentation = build_pose_presentation(raw, output_scope='full', resolve_library=asset)
        results.append((raw, presentation))
    assert normalize(results[0][1].model_dump()) == normalize(results[1][1].model_dump())
    for key in ('status', 'reason', 'selected', 'generation'):
        assert normalize(results[0][0].get(key)) == normalize(results[1][0].get(key))
