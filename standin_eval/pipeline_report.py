"""Join search/refine evidence to final-file tasks without changing denominators."""
from pathlib import Path

from .final_fbx_evaluator import evaluate_export_tasks
from .util import hash_json, read_json, read_jsonl, sha256_file


def build_pipeline_report(experiment, sealed_exports, observations):
    root = Path(experiment)
    manifest = read_json(root / 'manifest.json')
    if manifest.get('mode') != 'experiment':
        raise ValueError('an experiment run is required')
    required_results = {'experiment_arms.jsonl', 'experiment_report.json', 'search_evidence.json'}
    if not required_results <= set(manifest.get('result_seal', {})):
        raise ValueError('experiment result seal incomplete')
    for name, digest in manifest['result_seal'].items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents or sha256_file(path) != digest:
            raise ValueError('experiment result seal mismatch')
    if sha256_file(root / 'experiment.frozen.json') != manifest.get('frozen_sha256'):
        raise ValueError('experiment preregistration seal mismatch')
    frozen = read_json(root / 'experiment.frozen.json')
    units = read_jsonl(root / 'frozen_units.private.jsonl')
    if hash_json(units) != frozen.get('units_sha256'):
        raise ValueError('experiment selection seal mismatch')
    units = {r['unit_id']: r for r in units}
    rows = read_jsonl(root / 'experiment_arms.jsonl')
    by_key = {(r['unit_id'], r['arm']): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError('duplicate experiment unit/arm')
    exports = evaluate_export_tasks(sealed_exports, observations)
    observations_by_id = {r['task_id']: r for r in observations}
    links = []
    for task in sealed_exports['tasks']:
        failures, unknown = [], []
        expected = task.get('selection_keys')
        if not isinstance(expected, list) or not expected:
            unknown.append('export_task_selection_keys_unbound')
            expected_keys = set()
        else:
            expected_keys = {(r['unit_id'], r['arm']) for r in expected}
            if len(expected_keys) != len(expected):
                failures.append('duplicate_preregistered_selection')
        selections = observations_by_id.get(task['task_id'], {}).get('selections', [])
        actual_keys = {(s.get('unit_id'), s.get('arm')) for s in selections}
        if not selections:
            unknown.append('export_selection_not_measured')
        elif actual_keys != expected_keys or len(actual_keys) != len(selections):
            failures.append('export_selection_set_changed')
        for selection in selections:
            key = (selection.get('unit_id'), selection.get('arm'))
            row, unit = by_key.get(key), units.get(key[0])
            if row is None or unit is None:
                failures.append('export_has_no_experiment_artifact')
                continue
            if not row.get('result_bvh_sha256'):
                unknown.append('experiment_has_no_applied_artifact')
            elif selection.get('source_sha256') != row['result_bvh_sha256']:
                failures.append('export_uses_different_bvh')
            if row.get('state') not in {'pass', 'abstained'}:
                unknown.append('experiment_arm_not_delivered')
            for source_field, export_field in (('selection_revision', 'selection_revision'),
                    ('body_revision', 'body_revision'), ('hand_revision', 'hand_revision'),
                    ('output_scope', 'scope'), ('cut_id', 'cut_id'), ('gt_person_id', 'gt_person_id')):
                if not unit.get(source_field):
                    unknown.append('experiment_unbound:' + source_field)
                elif selection.get(export_field) != unit[source_field]:
                    failures.append('export_revision_mismatch:' + export_field)
            expected_camera = hash_json({'camera': unit.get('camera'), 'view': unit.get('view')})
            if selection.get('camera_sha256') != expected_camera:
                failures.append('export_camera_mismatch')
        links.append({'task_id': task['task_id'], 'state': 'fail' if failures else 'unknown' if unknown else 'pass',
                      'failures': sorted(set(failures)), 'unknown_reasons': sorted(set(unknown))})
    experiment_report = read_json(root / 'experiment_report.json')
    decision = ('FAIL' if 'FAIL' in {experiment_report['decision'], exports['decision']}
                or any(r['state'] == 'fail' for r in links) else 'INCONCLUSIVE'
                if any(r['state'] != 'pass' for r in links)
                or experiment_report['decision'] != 'PASS' or exports['decision'] != 'PASS' else 'PASS')
    return {'schema_version': 'evaluation-extension-v1', 'decision': decision,
            'experiment_run_id': manifest['run_id'], 'export_policy_sha256': sealed_exports['seal_sha256'],
            'search_refine': experiment_report, 'final_files': exports,
            'lineage_links': {'denominator': len(links),
                **{s: sum(r['state'] == s for r in links) for s in ('pass', 'fail', 'unknown')}, 'tasks': links},
            'operational_deployment': False}
