"""Bounded timing-only child diagnostics; never part of prepared-body identity.

Do not edit the six files hashed by prepared_body.identity just to add timings:
that would invalidate already approved neutral bodies. These delegates observe
the existing calls and preserve return values, exceptions and safety checks.
"""
from contextlib import contextmanager
import functools
import json
import math
import os
import time

PREFIX = 'STANDIN_BODY_TRACE '
STAGES = frozenset({
    'prepare_total', 'runtime_check', 'job_validation', 'body_worker',
    'neutral_restore', 'shape_source_load', 'shape_pose_transfer',
    'shape_controller_export', 'shape_controller_correct',
    'shape_controller_synchronize', 'shape_controller_compute',
    'shape_controller_geometry_validation', 'retarget_convert',
    'character_import', 'bvh_import', 'scene_reset', 'fbx_export',
    'output_mode', 'source_body', 'source_fingers', 'body_material',
    'final_fbx_validation', 'model_capture', 'body_snapshot_save',
    'model_encode', 'snapshot_write',
    'ankle_swing', 'ankle_clearance', 'foot_plant', 'airborne_plantar',
    'ankle_surface_measure', 'plantar_surface_measure', 'surface_positions',
    'surface_weights', 'leg_retarget',
})
REUSE_RESULTS = frozenset({'prepared', 'disabled', 'custom_shape', 'not_prepared', 'invalid_prepared'})


def validated_event(value):
    """Only a closed metrics schema can cross from Blender stdout to API logs."""
    if not isinstance(value, dict):
        return None
    if value.get('event') == 'converter_body_reuse':
        status = value.get('status')
        if status not in REUSE_RESULTS or type(value.get('hit')) is not bool:
            return None
        if value['hit'] != (status == 'prepared'):
            return None
        return dict(event='converter_body_reuse', status=status, hit=value['hit'])
    if value.get('event') == 'converter_body_solver':
        if value.get('stage') not in {'ankle_swing', 'ankle_clearance', 'foot_plant', 'airborne_plantar'}:
            return None
        if type(value.get('candidate_count')) is not int or not 0 <= value['candidate_count'] <= 10000:
            return None
        if type(value.get('fallback')) is not bool:
            return None
        return {k: value[k] for k in ('event', 'stage', 'candidate_count', 'fallback')}
    if value.get('event') != 'converter_body_stage' or value.get('stage') not in STAGES:
        return None
    if type(value.get('succeeded')) is not bool:
        return None
    for key in ('span_id', 'parent_span_id'):
        if type(value.get(key)) is not int or not 0 <= value[key] <= 10000:
            return None
    for key in ('start_ms', 'duration_ms'):
        if type(value.get(key)) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0:
            return None
    return {key: value[key] for key in ('event', 'stage', 'span_id', 'parent_span_id',
                                      'start_ms', 'duration_ms', 'succeeded')}


def relay_lines(raw, emit, request_id):
    """Drop ordinary Blender output and all unrecognized/private JSON fields."""
    for line in raw.splitlines()[:10000]:
        if not line.startswith(PREFIX) or len(line) > 4096:
            continue
        try:
            event = validated_event(json.loads(line[len(PREFIX):]))
        except (ValueError, TypeError):
            continue
        if event is not None:
            emit(json.dumps({**event, 'request_id': request_id}, separators=(',', ':')))


class BodyTrace:
    def __init__(self, enabled=None, clock=time.monotonic, emit=None):
        self.enabled = os.getenv('CONVERTER_BODY_TRACE_ENABLED', '1') == '1' if enabled is None else enabled
        self.clock = clock
        self.emit = emit or (lambda row: print(PREFIX + json.dumps(row, separators=(',', ':')), flush=True))
        self.started = clock()
        self.stack = []
        self.sequence = 0
        self.originals = []

    def publish(self, value):
        # Diagnostics must not turn a valid export into a failed operation.
        try:
            self.emit(value)
        except Exception:
            pass

    @contextmanager
    def span(self, stage):
        if not self.enabled:
            yield
            return
        start = self.clock()
        self.sequence += 1
        sid, parent = self.sequence, self.stack[-1] if self.stack else 0
        self.stack.append(sid)
        ok = False
        try:
            yield
            ok = True
        finally:
            self.stack.pop()
            self.publish(dict(event='converter_body_stage', stage=stage,
                span_id=sid, parent_span_id=parent, start_ms=round(max(0., start-self.started)*1000, 3),
                duration_ms=round(max(0., self.clock()-start)*1000, 3), succeeded=ok))

    def wrap(self, owner, name, stage, after=None):
        original = getattr(owner, name, None)
        if not callable(original):
            return
        @functools.wraps(original)
        def delegate(*args, **kwargs):
            with self.span(stage):
                result = original(*args, **kwargs)
                if after is not None:
                    try:
                        after(result)
                    except Exception:
                        pass
                return result
        self.originals.append((owner, name, original, delegate))
        setattr(owner, name, delegate)

    def controller(self, module):
        for name, stage in [('export_fixed', 'export'), ('correct_pose', 'correct'),
                ('synchronize', 'synchronize'), ('compute', 'compute'),
                ('validate_geometry', 'geometry_validation')]:
            self.wrap(module, name, 'shape_controller_' + stage)

    def reused(self, result):
        prototype, status = result
        self.publish(dict(event='converter_body_reuse', status=status, hit=prototype is not None))

    def solver_report(self, report):
        for stage in ('ankle_swing', 'ankle_clearance', 'foot_plant', 'airborne_plantar'):
            value = report.get(stage + '_selection')
            if not isinstance(value, dict):
                continue
            candidates = value.get('candidates', [])
            count = sum(len(rows) for rows in candidates.values() if isinstance(rows, list)) if isinstance(candidates, dict) else len(candidates)
            fallback = next((value[key] for key in ('fallback_to_exact_parent', 'fallback_to_exact_v321',
                'fallback_to_exact_v324') if type(value.get(key)) is bool), True)
            self.publish(dict(event='converter_body_solver', stage=stage,
                candidate_count=count, fallback=fallback))

    def __enter__(self):
        if not self.enabled:
            return self
        from converter import (interactive_adapter as adapter, interactive_scene as scene,
            worker, prepared_body, shape_bake, retarget, source_body, reviewed_source_fingers,
            body_material, final_artifact, solved_state_v2)
        from converter import convert
        from converter import ankle_swing, ankle_clearance, foot_plant, airborne_plantar
        for owner, name, stage, after in [
            (adapter, 'checked_runtime', 'runtime_check', None),
            (worker, 'load_job', 'job_validation', None),
            (adapter, '_run_body_job', 'body_worker', self.solver_report),
            (prepared_body, 'restore', 'neutral_restore', self.reused),
            (shape_bake, '_load', 'shape_source_load', self.controller),
            (shape_bake, 'transfer_solver_pose', 'shape_pose_transfer', None),
            (shape_bake, 'import_full_character', 'character_import', None),
            (convert, 'convert', 'retarget_convert', None),
            (retarget, 'import_bvh', 'bvh_import', None),
            (retarget, 'reset_scene', 'scene_reset', None),
            (retarget, 'export_fbx', 'fbx_export', None),
            (retarget, 'apply_output_mode', 'output_mode', None),
            (retarget, 'retarget', 'leg_retarget', None),
            (ankle_swing, 'convert_safe', 'ankle_swing', None),
            (ankle_clearance, 'convert_safe', 'ankle_clearance', None),
            (foot_plant, 'convert_safe', 'foot_plant', None),
            (airborne_plantar, 'convert_safe', 'airborne_plantar', None),
            (ankle_swing, '_surface_metrics', 'ankle_surface_measure', None),
            (ankle_swing, '_positions', 'surface_positions', None),
            (ankle_swing, '_vertex_weight_rows', 'surface_weights', None),
            (foot_plant, '_plantar_metrics', 'plantar_surface_measure', None),
            (source_body, 'bake_source_body', 'source_body', None),
            (reviewed_source_fingers, 'bake_source_fingers', 'source_fingers', None),
            (body_material, 'restore_anatomical_clay', 'body_material', None),
            (final_artifact, 'validate', 'final_fbx_validation', None),
            (scene, 'capture', 'model_capture', None),
            (adapter, '_save_scene', 'body_snapshot_save', None),
            (solved_state_v2, 'encode_model', 'model_encode', None),
            (adapter, 'atomic_write', 'snapshot_write', None),
        ]:
            self.wrap(owner, name, stage, after)
        return self

    def __exit__(self, *exc):
        for owner, name, original, delegate in reversed(self.originals):
            if getattr(owner, name) is delegate:
                setattr(owner, name, original)
