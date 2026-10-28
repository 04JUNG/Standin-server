"""Automatic COCO17 diagnostics on the fixed authored suite, offline by default.

PCK uses IMAGE DIAGONAL, not an inferred full-body bbox. Occluded GT estimates
are a separate diagnostic. Fixture/GT-derived results never certify a model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import html
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse

from .http_runner import _multipart, _request_json
from .matching import _hungarian
from .pose_dataset import load_pose_dataset
from .util import atomic_write_text, hash_json, percentile, read_json, read_jsonl, sha256_bytes, write_json, write_jsonl

POLICY = dict(schema_version='pose-recognition-policy-v1', pck_image_diagonal_fraction=.05,
              match_max_image_diagonal_fraction=.15, minimum_common_joints=1,
              prediction_score_threshold=None, latency_budget_ms=None,
              expected_inference_identity={},
              interpretation='engineering_diagnostic_not_product_acceptance')
IDENTITY_FIELDS = ('vlm_model','vlm_prompt_version','pose_model_version','deployment_version',
                   'pose_library_version','pose_library_sha256','feature_version','source_revision')


def _number(x):
    return type(x) in (int, float) and math.isfinite(x)


def _point(x):
    return isinstance(x, list) and len(x) == 2 and all(_number(v) for v in x)


def _validate_policy(policy):
    policy = copy.deepcopy(policy)
    if policy.get('schema_version') != POLICY['schema_version']:
        raise ValueError('unsupported recognition policy')
    for key in ('pck_image_diagonal_fraction', 'match_max_image_diagonal_fraction'):
        if not _number(policy.get(key)) or not 0 < policy[key] <= 1:
            raise ValueError('explicit normalized diagnostic thresholds required')
    if type(policy.get('minimum_common_joints')) is not int or not 1 <= policy['minimum_common_joints'] <= 17:
        raise ValueError('minimum_common_joints must be 1..17')
    for key in ('prediction_score_threshold', 'latency_budget_ms'):
        if policy.get(key) is not None and (not _number(policy[key]) or policy[key] < 0):
            raise ValueError('optional thresholds must be finite and nonnegative')
    if not isinstance(policy.get('expected_inference_identity', {}), dict):
        raise ValueError('expected inference identity must be a preregistered mapping')
    for key,value in policy.get('expected_inference_identity', {}).items():
        if key not in IDENTITY_FIELDS or not (isinstance(value,str) and value.strip() and value.lower() not in {'unknown','required','none'} or _number(value)):
            raise ValueError('inference identity requires resolved model/version/hash values')
    return policy


def _prediction_person(person, threshold):
    # Current /analyze output and simple COCO17 JSONL share this normalization.
    skeleton = person.get('skeleton') or {}
    points = person.get('keypoints', skeleton.get('keypoints'))
    scores = person.get('scores', skeleton.get('scores'))
    trace = person.get('quality_trace') or {}
    if not isinstance(trace, dict):
        raise ValueError('prediction quality trace must be an object')
    observed_mask = person.get('observed_joint_mask', trace.get('observed_joint_mask'))
    if observed_mask is not None and (not isinstance(observed_mask, list)
            or len(observed_mask) != 17 or any(type(v) is not bool for v in observed_mask)):
        raise ValueError('observation mask must contain 17 explicit booleans')
    if observed_mask is not None and 'observed_joint_scores' in trace:
        scores = trace['observed_joint_scores']
    if not isinstance(points, list) or len(points) != 17 or any(
            not isinstance(p, list) or len(p) != 2 or any(v is not None and not _number(v) for v in p) for p in points):
        raise ValueError('prediction requires 17 finite/null pixel points')
    if scores is not None and (not isinstance(scores, list) or len(scores) != 17
                               or any(v is not None and not _number(v) for v in scores)):
        raise ValueError('prediction score vector must have 17 finite/null values')
    return [p if _point(p) and (observed_mask is None or observed_mask[i])
            and (threshold is None or scores is not None and _number(scores[i]) and scores[i] >= threshold)
            else None for i, p in enumerate(points)]


def _association(gt_people, predictions, diagonal, policy):
    """Reuse the existing one-to-one solver; never assume array/personIndex IDs."""
    size = max(len(gt_people), len(predictions))
    if not size:
        return {}, {}, set()
    limit = policy['match_max_image_diagonal_fraction']
    costs = [[1.0] * size for _ in range(size)]
    detail = {}
    for i, gt in enumerate(gt_people):
        visible = [k for k, v in enumerate(gt['visibility']) if v == 2 and _point(gt['keypoints'][k])]
        basis = visible or [k for k, v in enumerate(gt['visibility']) if v == 1 and _point(gt['keypoints'][k])]
        for j, pred in enumerate(predictions):
            common = [k for k in basis if pred[k] is not None]
            distance = sum(math.dist(gt['keypoints'][k], pred[k]) / diagonal for k in common)/len(common) if common else None
            accepted = len(common) >= policy['minimum_common_joints'] and distance is not None and distance <= limit
            # Each accepted match costs < 1. All rejected edges cost 2, so a
            # low-error match can never force a much worse rejected assignment.
            costs[i][j] = distance / (limit * max(size, 1)) if accepted else 2.0
            detail[i, j] = dict(accepted=accepted, distance=distance, common_joints=len(common),
                                basis='visible' if visible else 'occluded_estimates_only' if basis else 'unavailable')
    assignment = _hungarian(costs)
    mapped, diagnostics, used = {}, {}, set()
    for i in range(len(gt_people)):
        j = assignment[i]
        evidence = detail.get((i, j))
        if evidence and evidence['accepted']:
            mapped[i], diagnostics[i] = j, evidence
            used.add(j)
    return mapped, diagnostics, used


def _bins(rows):
    counts = Counter(r['state'] for r in rows)
    return dict(denominator=len(rows), correct=counts['correct'], incorrect=counts['incorrect'],
                unknown=counts['unknown'], correct_rate=counts['correct']/len(rows) if rows and not counts['unknown'] else None,
                fixed_denominator_lower_bound=counts['correct']/len(rows) if rows else None)


def evaluate_pose_predictions(frozen, cuts, persons, observations, *, policy=None):
    policy = _validate_policy(policy or POLICY)
    by_cut, grouped = {}, defaultdict(list)
    allowed = {c['cut_id'] for c in cuts}
    for row in observations:
        cid = row.get('cut_id')
        if cid not in allowed or cid in by_cut:
            raise ValueError('prediction cut must be active and unique; exclusions cannot reenter')
        by_cut[cid] = row
    for p in persons:
        grouped[p['cut_id']].append(p)
    cut_rows, person_rows, joint_rows, hard_failures = [], [], [], []
    times = []
    for cut in cuts:
        cid = cut['cut_id']
        obs = by_cut.get(cid, {})
        state, reasons = obs.get('status', 'not_run'), []
        if state not in {'ok', 'error', 'timeout', 'not_run', 'unsupported', 'missing_dependency'}:
            state, reasons = 'error', ['invalid_prediction_status']
        identity = obs.get('image_sha256') == cut['image_sha256']
        if obs and not identity:
            state = 'error'
            reasons.append('prediction_image_identity_missing_or_mismatch')
            if obs.get('image_sha256'):
                hard_failures.append('prediction_image_mismatch:' + cid)
        requested, actual = obs.get('requested_backend') or {}, obs.get('actual_backend') or {}
        if not isinstance(requested, dict) or not isinstance(actual, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) or not v for values in (requested, actual) for k,v in values.items()):
            requested, actual = {}, {}
            reasons.append('backend_identity_malformed_or_incomplete')
        backend_state = 'unknown'
        if requested and all(actual.get(k) for k in requested):
            backend_state = 'pass' if all(actual[k] == v for k, v in requested.items()) else 'fail'
        if backend_state == 'fail':
            hard_failures.append('requested_actual_backend_mismatch:' + cid)
        raw_identity = obs.get('inference_metadata') or {}
        raw_identity = raw_identity if isinstance(raw_identity, dict) else {}
        identity_record = {k:v for k,v in raw_identity.items() if k in IDENTITY_FIELDS and
                           (isinstance(v,str) or _number(v))}
        expected_identity = policy.get('expected_inference_identity', {})
        model_identity_state = 'pass' if expected_identity and all(identity_record.get(k) == v for k,v in expected_identity.items()) else 'unknown'
        if any(k in identity_record and identity_record[k] != v for k,v in expected_identity.items()):
            model_identity_state = 'fail'
            hard_failures.append('inference_identity_mismatch:' + cid)
        kind = obs.get('execution_kind', 'not_executed')
        real = kind in {'http', 'recorded_model'} and not any(v in {'mock','fixture','replay','gt_contract_fixture'} for v in actual.values())
        gts = grouped[cid]
        predictions = []
        raw_people = obs.get('people', [])
        if state == 'ok':
            try:
                if not isinstance(raw_people, list):
                    raise ValueError('people must be a list')
                predictions = [_prediction_person(p, policy['prediction_score_threshold']) for p in raw_people]
            except (ValueError, TypeError, AttributeError):
                state = 'error'
                reasons.append('malformed_predicted_people')
        diagonal = math.hypot(cut['width'], cut['height'])
        mapped, association, used = _association(gts, predictions, diagonal, policy) if state == 'ok' else ({}, {}, set())
        count_state = 'unknown' if state in {'not_run','unsupported','missing_dependency'} or not identity else 'correct' if state == 'ok' and len(predictions) == cut['person_count'] else 'incorrect'
        vlm_count = obs.get('vlm_count')
        elapsed = obs.get('elapsed_ms')
        measured_time = real and backend_state == 'pass' and identity and state != 'not_run' and _number(elapsed) and elapsed >= 0
        time_state = ('unknown' if not measured_time or policy['latency_budget_ms'] is None else
                      'correct' if elapsed <= policy['latency_budget_ms'] else 'incorrect')
        cut_rows.append(dict(cut_id=cid, status=state, expected_persons=cut['person_count'],
            predicted_persons=len(predictions) if state == 'ok' else None, person_count_state=count_state,
            vlm_count=vlm_count if type(vlm_count) is int else None,
            vlm_count_state='unknown' if type(vlm_count) is not int or state != 'ok' else 'correct' if vlm_count == cut['person_count'] else 'incorrect',
            false_positives=len(predictions)-len(used) if state == 'ok' else None,
            gt_human_verified=cut['human_verified'], gt_evaluation_ready=cut['evaluation_ready'],
            requested_backend=requested, actual_backend=actual, backend_integrity=backend_state,
            reported_inference_identity=identity_record, model_identity_state=model_identity_state,
            time_state=time_state,
            execution_kind=kind, real_model_observation=real and backend_state == 'pass' and identity and state != 'not_run', reasons=reasons))
        for i, gt in enumerate(gts):
            visible = [k for k, v in enumerate(gt['visibility']) if v == 2 and _point(gt['keypoints'][k])]
            pending = state in {'not_run','unsupported','missing_dependency'} or not identity
            person_state = 'unknown' if pending or not visible else 'matched' if i in mapped else 'missed'
            person_rows.append(dict(cut_id=cid, person_id=gt['person_id'], state=person_state,
                prediction_index=mapped.get(i), association=association.get(i),
                known_visible_joints=len(visible), gt_missing=gt['gt_missing']))
            for k, visibility in enumerate(gt['visibility']):
                if visibility == 0:
                    continue
                pred = predictions[mapped[i]][k] if i in mapped else None
                valid_gt = _point(gt['keypoints'][k])
                error = math.dist(gt['keypoints'][k], pred) if valid_gt and pred is not None else None
                joint_state = 'unknown' if pending or not valid_gt else 'correct' if error is not None and error/diagonal <= policy['pck_image_diagonal_fraction'] else 'incorrect'
                joint_rows.append(dict(cut_id=cid, person_id=gt['person_id'], joint_index=k,
                    joint_name=frozen['keypoint_names'][k], visibility=visibility,
                    scope='visible_primary' if visibility == 2 else 'occluded_estimate_diagnostic',
                    state=joint_state, error_px=error, error_image_diagonal=error/diagonal if error is not None else None))
        if measured_time:
            times.append(elapsed)
    visible_rows = [r for r in joint_rows if r['visibility'] == 2]
    occluded_rows = [r for r in joint_rows if r['visibility'] == 1]
    def metric(rows):
        errors = [r['error_px'] for r in rows if r['error_px'] is not None]
        return {**_bins(rows), 'mean_error_px_on_comparable_joints':sum(errors)/len(errors) if errors else None,
                'comparable_joints':len(errors)}
    for row in cut_rows:
        row['visible_primary'] = metric([r for r in visible_rows if r['cut_id'] == row['cut_id']])
    report = dict(schema_version='pose-recognition-report-v1', dataset_id=frozen['dataset_id'],
        source_commit=frozen['source_commit'], decision='FAIL' if hard_failures else 'INCONCLUSIVE',
        score_status='PROVISIONAL_DRAFT_GT' if not all(c['human_verified'] and c['evaluation_ready'] for c in cuts) else 'ENGINEERING_ONLY',
        denominator=dict(cuts=len(cuts), persons=len(persons), visible_joints=len(visible_rows),
                         occluded_estimated_joints=len(occluded_rows), ignored_visibility_zero=sum(p['visibility'].count(0) for p in persons)),
        cut_statuses=dict(Counter(r['status'] for r in cut_rows)),
        person_association=dict(Counter(r['state'] for r in person_rows)),
        person_count=_bins([{'state':r['person_count_state']} for r in cut_rows]),
        vlm_person_count=_bins([{'state':r['vlm_count_state']} for r in cut_rows]),
        visible_primary=metric(visible_rows), occluded_estimates=metric(occluded_rows),
        visible_body_only=metric([r for r in visible_rows if r['joint_index'] >= 5]),
        by_joint={name:metric([r for r in visible_rows if r['joint_name']==name]) for name in frozen['keypoint_names']},
        policy=policy, hard_failures=sorted(set(hard_failures)),
        actual_model_cuts=sum(r['real_model_observation'] for r in cut_rows),
        model_identity_states=dict(Counter(r['model_identity_state'] for r in cut_rows)),
        latency=dict(kind='recorded_request_wall_clock_not_native_save', measured=len(times),
                     p50_ms=percentile(times,.5),p95_ms=percentile(times,.95),budget_ms=policy['latency_budget_ms'],
                     within_budget=_bins([{'state':r['time_state']} for r in cut_rows])),
        excluded_cut_ids=frozen['excluded_cut_ids'], human_verified_cuts=sum(c['human_verified'] for c in cuts),
        limitations=['no_product_quality_or_mesh_or_FBX_usability_certification',
                     'keypoint_matching_uses_GT_visible_support_not_human_verified_person_boxes',
                     'PCK_is_image_diagonal_normalized_not_bbox_or_head_normalized',
                     'occluded_coordinates_are_estimates_not_observed_truth',
                     'missing_outputs_and_unexecuted_cuts_keep_the_fixed_denominator'],
        operational_deployment=False)
    return report, cut_rows, person_rows, joint_rows


def run_pose_benchmark(dataset, output, *, predictions=None, policy=None, target=None,
                       allow_http=False, timeout_seconds=None, requested_backend=None):
    frozen, cuts, persons = load_pose_dataset(dataset)
    policy = _validate_policy(read_json(policy) if policy and not isinstance(policy, dict) else policy or POLICY)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new benchmark run output')
    if predictions and target:
        raise ValueError('choose recorded predictions or an explicit HTTP target')
    if target and (not allow_http or not _number(timeout_seconds) or timeout_seconds <= 0 or not requested_backend
                   or urllib.parse.urlparse(target).scheme not in {'http','https'}):
        raise ValueError('HTTP requires explicit allow_http, timeout and requested backend identities')
    if target and (set(requested_backend) != {'vlm','pose'} or not all(isinstance(v,str) and v for v in requested_backend.values())):
        raise ValueError('HTTP requires both requested VLM and pose backends')
    plan = dict(schema_version='pose-recognition-run-v1', dataset_seal=frozen['seal_sha256'],
        policy=policy, expected_cut_ids=[c['cut_id'] for c in cuts], expected_person_ids=[p['person_id'] for p in persons],
        input_predictions_sha256=sha256_bytes(Path(predictions).read_bytes()) if predictions else None,
        requested_backend=requested_backend, mode='http' if target else 'recorded' if predictions else 'not_executed')
    observations = read_jsonl(predictions) if predictions else []
    output.mkdir(parents=True)
    write_json(output/'plan.json', {**plan,'seal_sha256':hash_json(plan)})
    if target:
        (output/'responses').mkdir()
        for cut in cuts:
            started = time.monotonic()
            row = dict(cut_id=cut['cut_id'], image_sha256=cut['image_sha256'], execution_kind='http',
                       requested_backend=requested_backend, status='error', people=[])
            try:
                body, content_type = _multipart(Path(cut['image_path']))
                status, raw, payload = _request_json(urllib.parse.urljoin(target.rstrip('/')+'/', 'analyze'),
                    method='POST', data=body, content_type=content_type, timeout=timeout_seconds)
                atomic_write_text(output/'responses'/f"{cut['cut_id']}.json", raw.decode('utf-8',errors='replace'))
                if status == 200 and isinstance(payload, dict):
                    metadata = payload.get('inference_metadata') or {}
                    row.update(status='ok', people=payload.get('people'), vlm_count=payload.get('vlm_count'),
                        actual_backend={'vlm':metadata.get('vlm_provider'),'pose':metadata.get('pose_backend')},
                        inference_metadata={k:metadata.get(k) for k in IDENTITY_FIELDS})
            except (TimeoutError, urllib.error.URLError, OSError, ValueError, TypeError, AttributeError) as error:
                row['status'] = 'timeout' if isinstance(error, TimeoutError) or isinstance(getattr(error,'reason',None),TimeoutError) else 'error'
            row['elapsed_ms']=(time.monotonic()-started)*1000
            observations.append(row)
            write_jsonl(output/'observations.jsonl', observations)
    # Revalidate source after execution; mutable source cannot produce mixed GT.
    source_changed = False
    try:
        load_pose_dataset(dataset)
    except (ValueError, OSError):
        source_changed = True
    report, cut_rows, person_rows, joint_rows = evaluate_pose_predictions(frozen,cuts,persons,observations,policy=policy)
    if predictions and sha256_bytes(Path(predictions).read_bytes()) != plan['input_predictions_sha256']:
        report['hard_failures'].append('input_predictions_changed_during_scoring')
    if source_changed:
        report['hard_failures'].append('source_changed_during_execution')
    if report['hard_failures']:
        report['decision'] = 'FAIL'
    write_jsonl(output/'observations.jsonl',observations)
    for name, rows in [('cuts',cut_rows),('persons',person_rows),('joints',joint_rows)]:
        write_jsonl(output/(name+'.jsonl'),rows)
    report['plan_sha256']=hash_json(plan)
    write_json(output/'report.json',report)
    content = '<!doctype html><meta charset="utf-8"><title>Set pose benchmark</title><style>body{font:16px system-ui;margin:32px;color:#20344f}pre{white-space:pre-wrap}td,th{padding:8px;border-bottom:1px solid #ddd}</style>'
    content += '<h1>Pose recognition benchmark</h1><p>'+html.escape(report['decision']+' / '+report['score_status'])+'</p>'
    content += '<p>'+html.escape(str(report['denominator']))+'</p><p>Actual model observations: '+str(report['actual_model_cuts'])+'</p>'
    content += '<table><tr><th>Metric scope</th><th>Fixed denominator</th><th>Correct</th><th>Incorrect</th><th>Unknown</th><th>PCK</th><th>Comparable mean error (px)</th></tr>'
    for name in ['visible_primary','visible_body_only','occluded_estimates']:
        values=report[name]
        content += '<tr><td>'+html.escape(name)+'</td>'+''.join('<td>'+html.escape(str(values[k]))+'</td>' for k in
            ['denominator','correct','incorrect','unknown','correct_rate','mean_error_px_on_comparable_joints'])+'</tr>'
    content += '</table><p>Unknown inputs leave the aggregate PCK unavailable. Lower bounds and raw counts remain in report.json.</p>'
    content += '<p>Observed visible joints and occluded estimates are scored separately. No model ran unless explicitly recorded.</p><table><tr><th>Cut</th><th>Status</th><th>Expected persons</th><th>Predicted persons</th></tr>'
    content += ''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ['cut_id','status','expected_persons','predicted_persons'])+'</tr>' for r in cut_rows)+'</table>'
    atomic_write_text(output/'report.html',content)
    write_json(output/'manifest.json',dict(mode='pose_benchmark',plan_sha256=hash_json(plan),
        result_seal={n:sha256_bytes((output/n).read_bytes()) for n in ['report.json','report.html','cuts.jsonl','persons.jsonl','joints.jsonl','observations.jsonl']}))
    return report
